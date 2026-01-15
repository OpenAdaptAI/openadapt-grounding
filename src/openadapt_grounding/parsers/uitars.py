"""UI-TARS client for UI element grounding.

Connects to a UI-TARS vLLM server (OpenAI-compatible API) for element localization.
See: https://github.com/bytedance/UI-TARS
"""

import base64
import io
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image

from openadapt_grounding.types import Element


# Constants from UI-TARS action_parser.py
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def _round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    """Returns smallest integer >= 'number' divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    """Returns largest integer <= 'number' divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """Smart resize matching UI-TARS/Qwen2.5-VL preprocessing.

    Rescales dimensions so:
    1. Both are divisible by 'factor'
    2. Total pixels within [min_pixels, max_pixels]
    3. Aspect ratio maintained as closely as possible

    Args:
        height: Original image height
        width: Original image width
        factor: Divisibility factor (default 28)
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels

    Returns:
        Tuple of (resized_height, resized_width)
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"Aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(int(height / beta), factor)
        w_bar = _floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(int(height * beta), factor)
        w_bar = _ceil_by_factor(int(width * beta), factor)

    return h_bar, w_bar


@dataclass
class GroundingResult:
    """Result of a grounding query."""

    found: bool
    x: Optional[float] = None  # Normalized 0-1
    y: Optional[float] = None  # Normalized 0-1
    confidence: float = 0.0
    raw_response: str = ""
    thought: str = ""

    def to_pixels(self, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized coords to pixel coordinates.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Tuple of (x_pixels, y_pixels)

        Raises:
            ValueError: If element was not found
        """
        if not self.found or self.x is None or self.y is None:
            raise ValueError("Element not found")
        return int(self.x * width), int(self.y * height)


# Prompt templates from UI-TARS
GROUNDING_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
Action: ...

## Action Space
click(point='<point>x1 y1</point>')

## User Instruction
{instruction}"""

COMPUTER_USE_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c')
type(content='xxx')
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
wait()
finished(content='xxx')

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}"""


class UITarsClient:
    """Client for UI-TARS grounding via vLLM OpenAI-compatible API.

    Example:
        >>> client = UITarsClient("http://localhost:8001/v1")
        >>> if client.is_available():
        ...     result = client.ground(screenshot, "Click on the Login button")
        ...     if result.found:
        ...         px, py = result.to_pixels(1920, 1080)
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8001/v1",
        api_key: str = "not-needed",
        model: str = "ByteDance-Seed/UI-TARS-1.5-7B",
        timeout: float = 120.0,
    ):
        """Initialize the UI-TARS client.

        Args:
            server_url: URL of the vLLM server's OpenAI API endpoint
            api_key: API key (vLLM doesn't require auth by default)
            model: Model ID to use
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai not installed. Run: pip install openai"
                )
            self._client = OpenAI(
                base_url=self.server_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def is_available(self) -> bool:
        """Check if the UI-TARS server is available.

        Returns:
            True if server responds to health endpoint
        """
        try:
            import requests
            # vLLM health endpoint is at the base URL, not /v1
            base_url = self.server_url.replace("/v1", "")
            response = requests.get(f"{base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def ground(
        self,
        image: Image.Image,
        instruction: str,
        language: str = "English",
        include_thought: bool = False,
    ) -> GroundingResult:
        """Ground an element in the image by instruction.

        Args:
            image: Screenshot to search
            instruction: What to find (e.g., "Click on the Login button")
            language: Language for model output
            include_thought: If True, use COMPUTER_USE prompt with reasoning

        Returns:
            GroundingResult with normalized coordinates
        """
        # Encode image
        b64_image = self._image_to_base64(image)

        # Build prompt
        if include_thought:
            prompt = COMPUTER_USE_PROMPT.format(
                language=language,
                instruction=instruction,
            )
        else:
            prompt = GROUNDING_PROMPT.format(instruction=instruction)

        # Call model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.0,
        )

        raw_response = response.choices[0].message.content or ""

        # Parse response
        return self._parse_response(raw_response, image.size)

    def _parse_response(
        self,
        raw_response: str,
        image_size: Tuple[int, int],
    ) -> GroundingResult:
        """Parse UI-TARS response and extract coordinates.

        Args:
            raw_response: Raw model output
            image_size: Original image (width, height)

        Returns:
            GroundingResult with normalized coordinates
        """
        # Extract thought if present
        thought = ""
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", raw_response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract point coordinates
        # Format 1: click(point='<point>x y</point>')
        # Format 2: click(start_box='(x,y)') - UI-TARS 1.5 format
        point_match = re.search(r"<point>(\d+)\s+(\d+)</point>", raw_response)
        if not point_match:
            # Try start_box format
            point_match = re.search(r"start_box=['\"]?\((\d+),\s*(\d+)\)['\"]?", raw_response)
        if not point_match:
            return GroundingResult(
                found=False,
                raw_response=raw_response,
                thought=thought,
            )

        # Parse raw coordinates
        raw_x = int(point_match.group(1))
        raw_y = int(point_match.group(2))

        # Convert to normalized coordinates using smart_resize
        w, h = image_size
        resized_h, resized_w = smart_resize(h, w)

        # Normalize
        norm_x = raw_x / resized_w
        norm_y = raw_y / resized_h

        # Clamp to [0, 1]
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        return GroundingResult(
            found=True,
            x=norm_x,
            y=norm_y,
            confidence=0.9,  # UI-TARS doesn't provide confidence scores
            raw_response=raw_response,
            thought=thought,
        )

    def parse(self, image: Image.Image) -> List[Element]:
        """Parse all UI elements in the image.

        Note: UI-TARS is optimized for grounding (finding specific elements),
        not enumeration. This method exists for API compatibility but is
        not the recommended use case. Use OmniParser for full element parsing.

        Args:
            image: PIL Image of the screenshot

        Returns:
            List of Element objects (typically just one if found)
        """
        # For compatibility, we can ask UI-TARS to describe what it sees
        # But this is not its intended use case
        result = self.ground(image, "Describe all UI elements visible on screen")

        if result.found and result.x is not None and result.y is not None:
            # Return single element at the identified point
            # Since UI-TARS returns points, not bboxes, we create a small bbox
            return [
                Element(
                    bounds=(result.x - 0.025, result.y - 0.025, 0.05, 0.05),
                    text=result.thought or "detected element",
                    element_type="unknown",
                    confidence=result.confidence,
                )
            ]
        return []

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

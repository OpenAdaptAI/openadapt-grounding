"""OmniParser client for UI element detection.

Connects to an OmniParser v2 FastAPI server to extract UI elements from screenshots.
See: https://github.com/microsoft/OmniParser
"""

import base64
import io
from typing import List, Optional

import requests
from PIL import Image

from openadapt_grounding.types import Element


class OmniParserClient:
    """Client for OmniParser v2 FastAPI server.

    Example:
        >>> client = OmniParserClient("http://localhost:8000")
        >>> if client.is_available():
        ...     elements = client.parse(screenshot)
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: float = 60.0,
    ):
        """Initialize the OmniParser client.

        Args:
            server_url: URL of the OmniParser FastAPI server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if the OmniParser server is available.

        Returns:
            True if server responds to probe endpoint
        """
        try:
            response = requests.get(
                f"{self.server_url}/probe/",
                timeout=5.0,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def parse(self, image: Image.Image) -> List[Element]:
        """Parse a screenshot and return detected UI elements.

        Args:
            image: PIL Image of the screenshot

        Returns:
            List of Element objects with normalized bounds

        Raises:
            ConnectionError: If server is not available
            RuntimeError: If parsing fails
        """
        # Convert image to base64
        base64_image = self._image_to_base64(image)

        # Make request
        try:
            response = requests.post(
                f"{self.server_url}/parse/",
                json={"base64_image": base64_image},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to OmniParser server at {self.server_url}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OmniParser request failed: {e}") from e

        # Parse response
        result = response.json()
        return self._convert_response(result)

    def parse_with_metadata(self, image: Image.Image) -> dict:
        """Parse and return full response including metadata.

        Args:
            image: PIL Image of the screenshot

        Returns:
            Dict with 'elements', 'latency', and 'som_image' keys
        """
        base64_image = self._image_to_base64(image)

        response = requests.post(
            f"{self.server_url}/parse/",
            json={"base64_image": base64_image},
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        return {
            "elements": self._convert_response(result),
            "latency": result.get("latency", 0),
            "som_image_base64": result.get("som_image_base64"),
        }

    def _convert_response(self, result: dict) -> List[Element]:
        """Convert OmniParser response to Element list.

        OmniParser returns bboxes as [x1, y1, x2, y2] (normalized).
        We convert to (x, y, w, h) format.
        """
        elements = []
        parsed_content = result.get("parsed_content_list", [])

        for item in parsed_content:
            bbox = item.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            # Validate bounds
            if not all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
                continue

            # Convert [x1, y1, x2, y2] to (x, y, w, h)
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1

            # Skip invalid dimensions
            if w <= 0 or h <= 0:
                continue

            # Extract content/text
            content = item.get("content", "")

            # Determine element type from content prefix if present
            element_type = "unknown"
            text = content
            if content.startswith("Text Box ID"):
                element_type = "text"
                # Extract text after the ID prefix
                if ":" in content:
                    text = content.split(":", 1)[1].strip()
            elif content.startswith("Icon Box ID"):
                element_type = "icon"
                if ":" in content:
                    text = content.split(":", 1)[1].strip()

            elements.append(
                Element(
                    bounds=(x, y, w, h),
                    text=text,
                    element_type=element_type,
                    confidence=item.get("confidence", 1.0),
                )
            )

        return elements

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class OmniParserLocal:
    """Placeholder for local OmniParser integration.

    For running OmniParser locally without a server.
    Not yet implemented - requires torch and model weights.
    """

    def __init__(self, model_path: Optional[str] = None):
        raise NotImplementedError(
            "Local OmniParser not yet implemented. "
            "Use OmniParserClient with a deployed server instead. "
            "See: https://github.com/OpenAdaptAI/OpenAdapt/tree/main/deploy"
        )

"""Frame collection utilities for temporal smoothing."""

import time
from typing import Callable, List, Optional

from PIL import Image

from openadapt_grounding.builder import Registry, RegistryBuilder
from openadapt_grounding.parsers.base import Parser
from openadapt_grounding.types import Element


def collect_frames(
    parser: Parser,
    image: Image.Image,
    num_frames: int = 10,
    min_stability: float = 0.5,
    delay_ms: int = 0,
    on_frame: Optional[Callable[[int, List[Element]], None]] = None,
) -> Registry:
    """Collect multiple detection frames and build a stable registry.

    Runs the parser multiple times on the same image to capture detection
    variability (flickering), then filters to keep only stable elements.

    Args:
        parser: Parser instance (e.g., OmniParserClient)
        image: Screenshot to parse
        num_frames: Number of detection passes to run
        min_stability: Minimum fraction of frames an element must appear in (0-1)
        delay_ms: Delay between frames in milliseconds (for rate limiting)
        on_frame: Optional callback called after each frame with (frame_index, elements)

    Returns:
        Registry containing only stable elements

    Example:
        >>> client = OmniParserClient("http://localhost:8000")
        >>> screenshot = Image.open("screen.png")
        >>> registry = collect_frames(client, screenshot, num_frames=10)
        >>> print(f"Found {len(registry)} stable elements")
    """
    builder = RegistryBuilder()

    for i in range(num_frames):
        elements = parser.parse(image)
        builder.add_frame(elements)

        if on_frame:
            on_frame(i, elements)

        if delay_ms > 0 and i < num_frames - 1:
            time.sleep(delay_ms / 1000.0)

    return builder.build(min_stability=min_stability)


def collect_live_frames(
    parser: Parser,
    capture_fn: Callable[[], Image.Image],
    num_frames: int = 10,
    min_stability: float = 0.5,
    delay_ms: int = 100,
    on_frame: Optional[Callable[[int, Image.Image, List[Element]], None]] = None,
) -> Registry:
    """Collect frames from live screenshots for temporal smoothing.

    Unlike collect_frames(), this captures fresh screenshots between
    each detection pass, useful for detecting elements during live
    UI interaction.

    Args:
        parser: Parser instance (e.g., OmniParserClient)
        capture_fn: Function that returns a screenshot (e.g., mss capture)
        num_frames: Number of screenshots to capture and parse
        min_stability: Minimum fraction of frames an element must appear in
        delay_ms: Delay between captures in milliseconds
        on_frame: Optional callback with (frame_index, screenshot, elements)

    Returns:
        Registry containing only stable elements

    Example:
        >>> import mss
        >>> def capture():
        ...     with mss.mss() as sct:
        ...         return Image.frombytes("RGB", sct.grab(sct.monitors[0]).size, ...)
        >>> registry = collect_live_frames(client, capture, num_frames=10)
    """
    builder = RegistryBuilder()

    for i in range(num_frames):
        screenshot = capture_fn()
        elements = parser.parse(screenshot)
        builder.add_frame(elements)

        if on_frame:
            on_frame(i, screenshot, elements)

        if delay_ms > 0 and i < num_frames - 1:
            time.sleep(delay_ms / 1000.0)

    return builder.build(min_stability=min_stability)


def analyze_stability(
    parser: Parser,
    image: Image.Image,
    num_frames: int = 10,
) -> dict:
    """Analyze detection stability across multiple frames.

    Useful for understanding how reliable OmniParser is for a given screenshot.

    Args:
        parser: Parser instance
        image: Screenshot to analyze
        num_frames: Number of detection passes

    Returns:
        Dict with stability metrics per element and overall stats
    """
    all_frames: List[List[Element]] = []

    for _ in range(num_frames):
        elements = parser.parse(image)
        all_frames.append(elements)

    # Group elements by text (simple clustering)
    text_counts: dict[str, int] = {}
    text_bounds: dict[str, List[tuple]] = {}

    for frame in all_frames:
        for elem in frame:
            key = elem.text.lower().strip() if elem.text else ""
            if not key:
                continue

            text_counts[key] = text_counts.get(key, 0) + 1
            if key not in text_bounds:
                text_bounds[key] = []
            text_bounds[key].append(elem.bounds)

    # Calculate stability per element
    element_stats = []
    for text, count in text_counts.items():
        stability = count / num_frames
        bounds_list = text_bounds[text]

        # Calculate bounds variance
        if bounds_list:
            avg_x = sum(b[0] for b in bounds_list) / len(bounds_list)
            avg_y = sum(b[1] for b in bounds_list) / len(bounds_list)
            variance = sum(
                (b[0] - avg_x) ** 2 + (b[1] - avg_y) ** 2 for b in bounds_list
            ) / len(bounds_list)
        else:
            variance = 0

        element_stats.append(
            {
                "text": text,
                "detection_count": count,
                "stability": stability,
                "position_variance": variance,
            }
        )

    # Sort by stability (lowest first to highlight problems)
    element_stats.sort(key=lambda x: x["stability"])

    # Overall stats
    detection_counts = [len(f) for f in all_frames]
    avg_detections = sum(detection_counts) / len(detection_counts)
    stabilities = [e["stability"] for e in element_stats]

    return {
        "num_frames": num_frames,
        "unique_elements": len(text_counts),
        "avg_detections_per_frame": avg_detections,
        "min_detections": min(detection_counts),
        "max_detections": max(detection_counts),
        "avg_stability": sum(stabilities) / len(stabilities) if stabilities else 0,
        "elements": element_stats,
    }

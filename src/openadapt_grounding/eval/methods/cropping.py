"""Cropping strategies for evaluation methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image

from openadapt_grounding.eval.dataset.schema import AnnotatedElement
from openadapt_grounding.types import Element


@dataclass
class CropRegion:
    """Represents a crop region for evaluation."""

    x: float  # Normalized crop origin x
    y: float  # Normalized crop origin y
    w: float  # Normalized crop width
    h: float  # Normalized crop height

    def crop(self, image: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """Crop the image and return cropped image + pixel offset.

        Args:
            image: Original image to crop

        Returns:
            Tuple of (cropped_image, (px_x, px_y, px_w, px_h))
        """
        img_w, img_h = image.size
        px_x = int(self.x * img_w)
        px_y = int(self.y * img_h)
        px_w = int(self.w * img_w)
        px_h = int(self.h * img_h)

        # Ensure at least 1 pixel
        px_w = max(1, px_w)
        px_h = max(1, px_h)

        # Clamp to image bounds
        px_x = max(0, min(px_x, img_w - 1))
        px_y = max(0, min(px_y, img_h - 1))
        px_w = min(px_w, img_w - px_x)
        px_h = min(px_h, img_h - px_y)

        cropped = image.crop((px_x, px_y, px_x + px_w, px_y + px_h))
        return cropped, (px_x, px_y, px_w, px_h)

    def transform_element(
        self, elem: Element, offset: Tuple[int, int, int, int]
    ) -> Element:
        """Transform element coordinates from cropped space to original.

        Args:
            elem: Element detected in cropped image (normalized coords)
            offset: Pixel offset from crop (px_x, px_y, px_w, px_h)

        Returns:
            Element with coordinates in original image space
        """
        ex, ey, ew, eh = elem.bounds

        # Scale and translate
        orig_x = self.x + ex * self.w
        orig_y = self.y + ey * self.h
        orig_w = ew * self.w
        orig_h = eh * self.h

        return Element(
            bounds=(orig_x, orig_y, orig_w, orig_h),
            text=elem.text,
            element_type=elem.element_type,
            confidence=elem.confidence,
        )

    def transform_point(
        self, x: float, y: float, offset: Tuple[int, int, int, int]
    ) -> Tuple[float, float]:
        """Transform point from cropped space to original.

        Args:
            x: Normalized x coordinate in cropped image
            y: Normalized y coordinate in cropped image
            offset: Pixel offset (unused but kept for API consistency)

        Returns:
            Tuple of (orig_x, orig_y) in original image space
        """
        orig_x = self.x + x * self.w
        orig_y = self.y + y * self.h
        return (orig_x, orig_y)


class CroppingStrategy(ABC):
    """Abstract base for cropping strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""
        ...

    @abstractmethod
    def get_regions(
        self,
        image: Image.Image,
        target: AnnotatedElement,
    ) -> List[CropRegion]:
        """Get list of regions to evaluate.

        Args:
            image: Full screenshot
            target: Target element (may be used for adaptive cropping)

        Returns:
            List of CropRegion objects to evaluate
        """
        ...


class NoCropping(CroppingStrategy):
    """Baseline: evaluate on full image only."""

    @property
    def name(self) -> str:
        return "baseline"

    def get_regions(
        self, image: Image.Image, target: AnnotatedElement
    ) -> List[CropRegion]:
        return [CropRegion(0.0, 0.0, 1.0, 1.0)]


class FixedCropping(CroppingStrategy):
    """Fixed-size crops centered on expected location.

    For evaluation, we center crops on the target element to simulate
    what a real system might do with an approximate location.
    """

    def __init__(self, crop_sizes: List[int] = None):
        """Initialize with crop sizes.

        Args:
            crop_sizes: Sizes in pixels at 1080p resolution.
                        Converted to normalized during use.
        """
        self.crop_sizes = crop_sizes or [200, 300, 500]

    @property
    def name(self) -> str:
        return "fixed_crop"

    def get_regions(
        self, image: Image.Image, target: AnnotatedElement
    ) -> List[CropRegion]:
        regions = [CropRegion(0.0, 0.0, 1.0, 1.0)]  # Always include full image

        img_w, img_h = image.size
        cx, cy = target.click_point

        for size in self.crop_sizes:
            # Convert pixel size to normalized
            norm_w = size / img_w
            norm_h = size / img_h

            # Center crop on target
            crop_x = cx - norm_w / 2
            crop_y = cy - norm_h / 2

            # Clamp to image bounds
            crop_x = max(0.0, min(crop_x, 1.0 - norm_w))
            crop_y = max(0.0, min(crop_y, 1.0 - norm_h))

            # Ensure valid region
            if norm_w > 0 and norm_h > 0:
                regions.append(CropRegion(crop_x, crop_y, norm_w, norm_h))

        return regions


class ScreenSeekeRCropping(CroppingStrategy):
    """LLM-guided progressive cropping (ScreenSeekeR-style).

    Uses heuristic UI regions when no LLM is available.
    """

    def __init__(self, llm_client=None, max_regions: int = 5):
        """Initialize cropping strategy.

        Args:
            llm_client: Optional LLM client for region prediction
            max_regions: Maximum number of regions to return
        """
        self.llm_client = llm_client
        self.max_regions = max_regions

    @property
    def name(self) -> str:
        return "screenseeker"

    def get_regions(
        self, image: Image.Image, target: AnnotatedElement
    ) -> List[CropRegion]:
        regions = [CropRegion(0.0, 0.0, 1.0, 1.0)]  # Always include full image

        # If no LLM client, use heuristic regions
        if self.llm_client is None:
            regions.extend(self._heuristic_regions(target))
            return regions[: self.max_regions + 1]

        # Use LLM to predict regions (future implementation)
        predicted_regions = self._llm_predict_regions(image, target)
        regions.extend(predicted_regions[: self.max_regions])

        return regions

    def _heuristic_regions(self, target: AnnotatedElement) -> List[CropRegion]:
        """Fallback heuristic regions based on common UI patterns."""
        cx, cy = target.click_point
        regions = []

        # Top toolbar region
        if cy < 0.15:
            regions.append(CropRegion(0.0, 0.0, 1.0, 0.15))

        # Left sidebar region
        if cx < 0.25:
            regions.append(CropRegion(0.0, 0.0, 0.3, 1.0))

        # Right panel region
        if cx > 0.75:
            regions.append(CropRegion(0.7, 0.0, 0.3, 1.0))

        # Bottom status bar region
        if cy > 0.9:
            regions.append(CropRegion(0.0, 0.85, 1.0, 0.15))

        # Center content region (always include)
        regions.append(CropRegion(0.15, 0.1, 0.7, 0.8))

        # Quadrant containing target
        quad_x = 0.0 if cx < 0.5 else 0.5
        quad_y = 0.0 if cy < 0.5 else 0.5
        regions.append(CropRegion(quad_x, quad_y, 0.5, 0.5))

        return regions

    def _llm_predict_regions(
        self,
        image: Image.Image,
        target: AnnotatedElement,
    ) -> List[CropRegion]:
        """Use LLM to predict likely regions containing the target.

        Future implementation would:
        1. Send image to LLM with target description
        2. Ask LLM to identify likely UI regions
        3. Parse response into CropRegion objects
        """
        # Placeholder for LLM-based region prediction
        return self._heuristic_regions(target)

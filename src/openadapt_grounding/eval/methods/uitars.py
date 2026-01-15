"""UI-TARS evaluation method."""

import time
from typing import Optional

from PIL import Image

from openadapt_grounding.eval.dataset.schema import AnnotatedElement
from openadapt_grounding.eval.methods.base import EvaluationMethod, EvaluationPrediction
from openadapt_grounding.eval.methods.cropping import CroppingStrategy, NoCropping
from openadapt_grounding.parsers.uitars import UITarsClient


class UITarsMethod(EvaluationMethod):
    """UI-TARS-based evaluation method.

    Strategy: Use instruction to ground element, check if point falls within target bbox.
    """

    def __init__(
        self,
        client: UITarsClient,
        cropping: Optional[CroppingStrategy] = None,
        bbox_tolerance: float = 0.02,
    ):
        """Initialize UI-TARS evaluation method.

        Args:
            client: UI-TARS client instance
            cropping: Cropping strategy to use
            bbox_tolerance: Extra tolerance around bbox for point matching
        """
        self.client = client
        self.cropping = cropping or NoCropping()
        self.bbox_tolerance = bbox_tolerance

    @property
    def name(self) -> str:
        return f"UI-TARS + {self.cropping.name}"

    def is_available(self) -> bool:
        return self.client.is_available()

    def evaluate_element(
        self,
        image: Image.Image,
        target_element: AnnotatedElement,
    ) -> EvaluationPrediction:
        """Evaluate detection of a single element.

        Args:
            image: Screenshot to evaluate
            target_element: Ground truth element to find

        Returns:
            EvaluationPrediction with found status, coordinates, and timing
        """
        start_time = time.perf_counter()
        attempts = 0

        # Build instruction from element
        instruction = self._build_instruction(target_element)

        # Get cropped regions to evaluate
        regions = self.cropping.get_regions(image, target_element)

        for region in regions:
            attempts += 1
            cropped_image, offset = region.crop(image)

            # Ground element in cropped region
            try:
                result = self.client.ground(cropped_image, instruction)
            except Exception:
                continue

            if result.found and result.x is not None and result.y is not None:
                # Transform coordinates back to original image space
                orig_x, orig_y = region.transform_point(result.x, result.y, offset)

                # Check if point is within target bbox (with tolerance)
                if self._point_in_bbox(orig_x, orig_y, target_element.bbox):
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return EvaluationPrediction(
                        found=True,
                        click_point=(orig_x, orig_y),
                        bbox=None,  # UI-TARS returns points, not bboxes
                        confidence=result.confidence,
                        latency_ms=latency_ms,
                        attempts=attempts,
                        method_info={
                            "thought": result.thought,
                            "instruction": instruction,
                        },
                    )

        latency_ms = (time.perf_counter() - start_time) * 1000
        return EvaluationPrediction(
            found=False,
            latency_ms=latency_ms,
            attempts=attempts,
            method_info={"instruction": instruction},
        )

    def _build_instruction(self, element: AnnotatedElement) -> str:
        """Build grounding instruction from element annotation.

        Args:
            element: Annotated element with text and type info

        Returns:
            Natural language instruction for UI-TARS
        """
        if element.instruction:
            return element.instruction

        # Generate instruction from element properties
        elem_type = element.element_type.value
        if element.text:
            if elem_type == "unknown":
                return f"Click on '{element.text}'"
            return f"Click on the '{element.text}' {elem_type}"
        else:
            return f"Click on the {elem_type}"

    def _point_in_bbox(
        self, x: float, y: float, bbox: tuple, tolerance: Optional[float] = None
    ) -> bool:
        """Check if point (x, y) is within bbox (bx, by, bw, bh).

        Args:
            x: Normalized x coordinate
            y: Normalized y coordinate
            bbox: Bounding box (x, y, w, h)
            tolerance: Extra tolerance around bbox edges

        Returns:
            True if point is within bbox
        """
        tol = tolerance if tolerance is not None else self.bbox_tolerance
        bx, by, bw, bh = bbox

        return (bx - tol) <= x <= (bx + bw + tol) and (by - tol) <= y <= (by + bh + tol)

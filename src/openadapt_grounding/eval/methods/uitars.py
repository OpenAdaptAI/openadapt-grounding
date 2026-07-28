"""UI-TARS evaluation method."""

import logging
import time
from typing import Any

from PIL import Image

from openadapt_grounding.eval.dataset.schema import AnnotatedElement
from openadapt_grounding.eval.methods.base import EvaluationMethod, EvaluationPrediction
from openadapt_grounding.eval.methods.cropping import CroppingStrategy, NoCropping
from openadapt_grounding.parsers.uitars import UITarsClient

logger = logging.getLogger(__name__)


class UITarsMethod(EvaluationMethod):
    """UI-TARS-based evaluation method.

    Strategy: Use instruction to ground element, check if point falls within target bbox.
    """

    def __init__(
        self,
        client: UITarsClient,
        cropping: CroppingStrategy | None = None,
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

        # Grounding failures are recorded, not swallowed. Dropping them
        # silently made an unreachable or erroring UI-TARS server
        # indistinguishable from a genuine miss, so a backend outage was
        # published as "0% accuracy" in the method comparison instead of an
        # invalid run.
        errors: list[str] = []

        for region in regions:
            attempts += 1
            cropped_image, offset = region.crop(image)

            # Ground element in cropped region
            try:
                result = self.client.ground(cropped_image, instruction)
            except Exception as exc:
                logger.warning(
                    "UI-TARS failed on region %d of %d for element %s: %s",
                    attempts,
                    len(regions),
                    target_element.id,
                    exc,
                )
                errors.append(f"{type(exc).__name__}: {exc}")
                continue

            if result.found and result.x is not None and result.y is not None:
                # Transform coordinates back to original image space
                orig_x, orig_y = region.transform_point(result.x, result.y, offset)

                # Check if point is within target bbox (with tolerance)
                if self._point_in_bbox(orig_x, orig_y, target_element.bbox):
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    hit_info: dict[str, Any] = {
                        "thought": result.thought,
                        "instruction": instruction,
                    }
                    if errors:
                        hit_info["region_errors"] = errors
                    return EvaluationPrediction(
                        found=True,
                        click_point=(orig_x, orig_y),
                        bbox=None,  # UI-TARS returns points, not bboxes
                        confidence=result.confidence,
                        latency_ms=latency_ms,
                        attempts=attempts,
                        method_info=hit_info,
                    )

        latency_ms = (time.perf_counter() - start_time) * 1000
        method_info: dict[str, Any] = {"instruction": instruction}
        if errors:
            # Carried into the stored results JSON so a "not found" caused by a
            # broken backend is visible after the fact.
            method_info["region_errors"] = errors
        return EvaluationPrediction(
            found=False,
            latency_ms=latency_ms,
            attempts=attempts,
            method_info=method_info,
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
        self, x: float, y: float, bbox: tuple, tolerance: float | None = None
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

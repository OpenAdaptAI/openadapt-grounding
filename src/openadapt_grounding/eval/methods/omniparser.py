"""OmniParser evaluation method."""

import time
from typing import Optional

from PIL import Image

from openadapt_grounding.eval.dataset.schema import AnnotatedElement
from openadapt_grounding.eval.methods.base import EvaluationMethod, EvaluationPrediction
from openadapt_grounding.eval.methods.cropping import CroppingStrategy, NoCropping
from openadapt_grounding.parsers.omniparser import OmniParserClient
from openadapt_grounding.types import Element


class OmniParserMethod(EvaluationMethod):
    """OmniParser-based evaluation method.

    Strategy: Parse image, find element that best matches target.

    Matching criteria:
    1. Text match (if target has text)
    2. Bbox overlap (IoU)
    3. Distance from target click point
    """

    def __init__(
        self,
        client: OmniParserClient,
        cropping: Optional[CroppingStrategy] = None,
        iou_threshold: float = 0.3,
        text_match_threshold: float = 0.8,
    ):
        """Initialize OmniParser evaluation method.

        Args:
            client: OmniParser client instance
            cropping: Cropping strategy to use
            iou_threshold: Minimum IoU to consider a match
            text_match_threshold: Minimum text similarity for text match
        """
        self.client = client
        self.cropping = cropping or NoCropping()
        self.iou_threshold = iou_threshold
        self.text_match_threshold = text_match_threshold

    @property
    def name(self) -> str:
        return f"OmniParser + {self.cropping.name}"

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

        # Get cropped regions to evaluate
        regions = self.cropping.get_regions(image, target_element)

        best_match: Optional[Element] = None
        best_score = 0.0

        for region in regions:
            attempts += 1
            cropped_image, offset = region.crop(image)

            # Parse cropped region
            try:
                elements = self.client.parse(cropped_image)
            except Exception:
                continue

            # Find best matching element
            for elem in elements:
                # Transform coordinates back to original image space
                transformed_elem = region.transform_element(elem, offset)

                score = self._compute_match_score(transformed_elem, target_element)
                if score > best_score:
                    best_score = score
                    best_match = transformed_elem

            # Early exit if we found a good match
            if best_score >= 0.9:
                break

        latency_ms = (time.perf_counter() - start_time) * 1000

        if best_match is not None and best_score >= self.iou_threshold:
            return EvaluationPrediction(
                found=True,
                click_point=best_match.center,
                bbox=best_match.bounds,
                confidence=best_score,
                latency_ms=latency_ms,
                attempts=attempts,
                method_info={"matched_text": best_match.text, "score": best_score},
            )

        return EvaluationPrediction(
            found=False,
            latency_ms=latency_ms,
            attempts=attempts,
            method_info={"best_score": best_score},
        )

    def _compute_match_score(
        self,
        detected: Element,
        target: AnnotatedElement,
    ) -> float:
        """Compute match score between detected and target element.

        Args:
            detected: Detected element from parser
            target: Ground truth annotated element

        Returns:
            Score between 0 and 1
        """
        # Create Element from target for IoU computation
        target_elem = Element(bounds=target.bbox, text=target.text)

        # IoU score
        iou = detected.iou(target_elem)

        # Text similarity bonus
        text_bonus = 0.0
        if detected.text and target.text:
            detected_text = detected.text.lower().strip()
            target_text = target.text.lower().strip()

            if detected_text == target_text:
                text_bonus = 0.3
            elif target_text in detected_text or detected_text in target_text:
                text_bonus = 0.15

        # Distance penalty for far matches
        det_cx, det_cy = detected.center
        tgt_cx, tgt_cy = target.click_point
        distance = ((det_cx - tgt_cx) ** 2 + (det_cy - tgt_cy) ** 2) ** 0.5
        distance_penalty = min(0.2, distance * 0.5)

        return min(1.0, max(0.0, iou + text_bonus - distance_penalty))

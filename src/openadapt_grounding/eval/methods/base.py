"""Abstract base for evaluation methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from openadapt_grounding.eval.dataset.schema import AnnotatedElement
from openadapt_grounding.types import Bounds


@dataclass
class EvaluationPrediction:
    """Unified prediction result for evaluation.

    This class unifies OmniParser (bbox-based) and UI-TARS (point-based)
    predictions into a common format for metrics computation.
    """

    found: bool
    click_point: Optional[Tuple[float, float]] = None  # Normalized (x, y)
    bbox: Optional[Bounds] = None  # (x, y, w, h) normalized
    confidence: float = 0.0
    latency_ms: float = 0.0
    attempts: int = 1
    method_info: Dict[str, Any] = field(default_factory=dict)


class EvaluationMethod(ABC):
    """Abstract base for all evaluation methods.

    This unifies OmniParser and UI-TARS under a common interface for fair comparison.

    For OmniParser: Parse image, find element that best matches the target
    For UI-TARS: Query with instruction, check if returned point is in target bbox
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable method name."""
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the method's backend is available."""
        ...

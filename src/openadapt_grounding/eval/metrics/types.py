"""Metric result types for evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ElementResult:
    """Result for a single element evaluation."""

    sample_id: str
    element_id: str
    found: bool
    iou: float  # IoU with ground truth (0 if UI-TARS point-based)
    latency_ms: float
    attempts: int
    size_category: str  # "small", "medium", "large"
    element_type: str
    distance_from_target: float  # Distance from prediction to target center
    method_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "sample_id": self.sample_id,
            "element_id": self.element_id,
            "found": self.found,
            "iou": self.iou,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "size_category": self.size_category,
            "element_type": self.element_type,
            "distance_from_target": self.distance_from_target,
            "method_info": self.method_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElementResult":
        """Create from dict."""
        return cls(
            sample_id=data["sample_id"],
            element_id=data["element_id"],
            found=data["found"],
            iou=data["iou"],
            latency_ms=data["latency_ms"],
            attempts=data["attempts"],
            size_category=data["size_category"],
            element_type=data["element_type"],
            distance_from_target=data["distance_from_target"],
            method_info=data.get("method_info", {}),
        )


@dataclass
class MethodMetrics:
    """Aggregated metrics for a single method on a dataset."""

    method_name: str
    dataset_name: str

    # Primary metrics
    detection_rate: float
    mean_iou: float
    mean_latency_ms: float
    mean_attempts: float

    # Breakdown by size
    detection_rate_small: float
    detection_rate_medium: float
    detection_rate_large: float

    # Breakdown by element type
    detection_rate_by_type: Dict[str, float]

    # Total counts
    total_elements: int
    detected_elements: int

    # Raw results for detailed analysis
    results: List[ElementResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "method_name": self.method_name,
            "dataset_name": self.dataset_name,
            "detection_rate": self.detection_rate,
            "mean_iou": self.mean_iou,
            "mean_latency_ms": self.mean_latency_ms,
            "mean_attempts": self.mean_attempts,
            "detection_rate_small": self.detection_rate_small,
            "detection_rate_medium": self.detection_rate_medium,
            "detection_rate_large": self.detection_rate_large,
            "detection_rate_by_type": self.detection_rate_by_type,
            "total_elements": self.total_elements,
            "detected_elements": self.detected_elements,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MethodMetrics":
        """Create from dict."""
        return cls(
            method_name=data["method_name"],
            dataset_name=data["dataset_name"],
            detection_rate=data["detection_rate"],
            mean_iou=data["mean_iou"],
            mean_latency_ms=data["mean_latency_ms"],
            mean_attempts=data["mean_attempts"],
            detection_rate_small=data["detection_rate_small"],
            detection_rate_medium=data["detection_rate_medium"],
            detection_rate_large=data["detection_rate_large"],
            detection_rate_by_type=data["detection_rate_by_type"],
            total_elements=data["total_elements"],
            detected_elements=data["detected_elements"],
            results=[ElementResult.from_dict(r) for r in data.get("results", [])],
        )

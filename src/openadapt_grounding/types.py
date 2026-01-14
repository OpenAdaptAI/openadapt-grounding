"""Core data types for openadapt-grounding."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Normalized bounds: (x, y, width, height) where all values are 0.0-1.0
Bounds = Tuple[float, float, float, float]


@dataclass
class Element:
    """A detected UI element from a single frame."""

    bounds: Bounds  # Normalized (x, y, w, h)
    text: Optional[str] = None
    element_type: str = "unknown"
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point in normalized coordinates."""
        x, y, w, h = self.bounds
        return (x + w / 2, y + h / 2)

    def iou(self, other: "Element") -> float:
        """Compute Intersection over Union with another element."""
        x1, y1, w1, h1 = self.bounds
        x2, y2, w2, h2 = other.bounds

        # Compute intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class RegistryEntry:
    """A stable element in the registry (survived temporal filtering)."""

    uid: str
    text: Optional[str]
    bounds: Bounds  # Average/representative bounds
    element_type: str
    detection_count: int  # How many frames it appeared in
    total_frames: int  # Total frames processed

    @property
    def stability(self) -> float:
        """Fraction of frames this element was detected in."""
        if self.total_frames == 0:
            return 0.0
        return self.detection_count / self.total_frames

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point in normalized coordinates."""
        x, y, w, h = self.bounds
        return (x + w / 2, y + h / 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "uid": self.uid,
            "text": self.text,
            "bounds": list(self.bounds),
            "element_type": self.element_type,
            "detection_count": self.detection_count,
            "total_frames": self.total_frames,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegistryEntry":
        """Create from dict."""
        return cls(
            uid=data["uid"],
            text=data.get("text"),
            bounds=tuple(data["bounds"]),  # type: ignore
            element_type=data.get("element_type", "unknown"),
            detection_count=data.get("detection_count", 1),
            total_frames=data.get("total_frames", 1),
        )


@dataclass
class LocatorResult:
    """Result of attempting to locate an element."""

    found: bool
    x: Optional[float] = None  # Normalized X coordinate
    y: Optional[float] = None  # Normalized Y coordinate
    confidence: float = 0.0
    matched_entry: Optional[RegistryEntry] = None
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_pixels(self, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Convert normalized coordinates to pixel coordinates."""
        if not self.found or self.x is None or self.y is None:
            return None
        return (int(self.x * width), int(self.y * height))

"""Dataset schema for evaluation annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ElementType(str, Enum):
    """Type of UI element."""

    BUTTON = "button"
    ICON = "icon"
    TEXT = "text"
    TEXT_FIELD = "text_field"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    LINK = "link"
    UNKNOWN = "unknown"


class ElementSize(str, Enum):
    """Size category based on screen percentage."""

    SMALL = "small"  # < 0.1% of screen (< 32px at 1080p)
    MEDIUM = "medium"  # 0.1% - 1% of screen (32-100px at 1080p)
    LARGE = "large"  # > 1% of screen (> 100px at 1080p)


@dataclass
class AnnotatedElement:
    """Ground truth annotation for a single UI element."""

    id: str
    bbox: Tuple[float, float, float, float]  # (x, y, w, h) normalized 0-1
    text: Optional[str] = None
    element_type: ElementType = ElementType.UNKNOWN
    instruction: Optional[str] = None  # Natural language description for UI-TARS

    @property
    def click_point(self) -> Tuple[float, float]:
        """Center point of the element."""
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)

    @property
    def size_category(self) -> ElementSize:
        """Categorize element by size."""
        _, _, w, h = self.bbox
        area = w * h
        if area < 0.001:  # < 0.1%
            return ElementSize.SMALL
        elif area < 0.01:  # < 1%
            return ElementSize.MEDIUM
        else:
            return ElementSize.LARGE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "id": self.id,
            "bbox": list(self.bbox),
            "text": self.text,
            "element_type": self.element_type.value,
            "instruction": self.instruction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnnotatedElement:
        """Create from dict."""
        return cls(
            id=data["id"],
            bbox=tuple(data["bbox"]),  # type: ignore
            text=data.get("text"),
            element_type=ElementType(data.get("element_type", "unknown")),
            instruction=data.get("instruction"),
        )


@dataclass
class Sample:
    """A single evaluation sample (screenshot + annotations)."""

    id: str
    image_path: str  # Relative path from dataset root
    width: int  # Original image width in pixels
    height: int  # Original image height in pixels
    elements: List[AnnotatedElement]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "id": self.id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "elements": [e.to_dict() for e in self.elements],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Sample:
        """Create from dict."""
        return cls(
            id=data["id"],
            image_path=data["image_path"],
            width=data["width"],
            height=data["height"],
            elements=[AnnotatedElement.from_dict(e) for e in data.get("elements", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class Dataset:
    """Complete evaluation dataset."""

    name: str
    version: str
    samples: List[Sample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save dataset to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "version": self.version,
            "metadata": self.metadata,
            "samples": [s.to_dict() for s in self.samples],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Dataset:
        """Load dataset from JSON."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            name=data["name"],
            version=data["version"],
            samples=[Sample.from_dict(s) for s in data.get("samples", [])],
            metadata=data.get("metadata", {}),
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def total_elements(self) -> int:
        """Return total number of annotated elements."""
        return sum(len(s.elements) for s in self.samples)

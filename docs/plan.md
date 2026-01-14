# OpenAdapt Grounding: Design & Implementation Plan

**Version:** 0.2.0 (Simplified)
**Date:** 2026-01-14
**Status:** Draft

---

## 1. Problem

OmniParser is flakey on single frames. We need reliable UI element localization.

## 2. Solution (80/20)

Two simple ideas that solve 80% of the problem:

1. **Temporal Smoothing**: Run detection on multiple frames, keep elements that persist
2. **Text Anchoring**: Match elements by their text content (resolution-independent)

## 3. Architecture

```
OFFLINE: Record → Detect (N frames) → Filter stable elements → Save Registry
ONLINE:  Screenshot → OCR/Match → Return (x,y) or ABSTAIN
```

## 4. Core API

```python
from openadapt_grounding import RegistryBuilder, ElementLocator

# Build registry from recording
builder = RegistryBuilder()
builder.add_frame(screenshot1, detections1)
builder.add_frame(screenshot2, detections2)
# ... add more frames
registry = builder.build(min_stability=0.5)  # Keep elements in >50% of frames
registry.save("elements.json")

# Locate at runtime
locator = ElementLocator("elements.json")
result = locator.find("Post Batch", screenshot)
if result:
    print(f"Found at ({result.x}, {result.y})")
```

## 5. Project Structure

```
openadapt-grounding/
├── pyproject.toml
├── README.md
├── src/openadapt_grounding/
│   ├── __init__.py
│   ├── types.py          # Simple dataclasses
│   ├── builder.py        # Registry construction
│   ├── locator.py        # Runtime element finding
│   ├── ocr.py            # Text detection
│   └── utils.py          # Screenshots, coordinates
└── tests/
    └── test_basic.py
```

## 6. Data Types

```python
@dataclass
class Element:
    text: Optional[str]
    bounds: Tuple[float, float, float, float]  # Normalized (x, y, w, h)
    element_type: str = "unknown"
    confidence: float = 1.0

@dataclass
class RegistryEntry:
    uid: str
    text: Optional[str]
    bounds: Tuple[float, float, float, float]
    detection_count: int
    total_frames: int

@dataclass
class LocatorResult:
    found: bool
    x: Optional[float] = None  # Normalized
    y: Optional[float] = None
    confidence: float = 0.0
```

## 7. Implementation

### 7.1 Registry Builder

```python
class RegistryBuilder:
    def __init__(self):
        self.frames = []

    def add_frame(self, elements: List[Element]):
        self.frames.append(elements)

    def build(self, min_stability: float = 0.5) -> Registry:
        # Group elements by text (primary) or location (fallback)
        # Keep only those appearing in > min_stability fraction of frames
        clusters = self._cluster_elements()
        stable = [c for c in clusters if c.stability >= min_stability]
        return Registry(stable)

    def _cluster_elements(self):
        # Simple: group by exact text match
        # For elements without text: group by IoU > 0.5
        ...
```

### 7.2 Element Locator

```python
class ElementLocator:
    def __init__(self, registry_path: str):
        self.registry = Registry.load(registry_path)

    def find(self, query: str, screenshot: Image) -> Optional[LocatorResult]:
        # 1. Try exact text match in registry
        entry = self.registry.get_by_text(query)
        if not entry:
            return LocatorResult(found=False)

        # 2. OCR the screenshot, find matching text
        ocr_results = self._ocr(screenshot)
        for result in ocr_results:
            if self._text_matches(result.text, entry.text):
                cx, cy = self._get_center(result.bounds)
                return LocatorResult(found=True, x=cx, y=cy, confidence=0.9)

        # 3. Fallback: check expected location (spatial prior)
        # If screen looks similar, trust the stored coordinates
        return LocatorResult(found=False)
```

## 8. Dependencies (Minimal)

```toml
dependencies = [
    "pillow>=10.0.0",
    "pytesseract>=0.3.10",  # OCR
    "requests>=2.31.0",      # OmniParser API (optional)
]
```

## 9. What We're NOT Building (Complexity Avoided)

- ❌ Multiple OCR backends (just Tesseract)
- ❌ Visual embeddings (OCR is enough for text elements)
- ❌ Unified multi-signal scorer (simple if/else is fine)
- ❌ Integration with openadapt-ml (standalone)
- ❌ AWS deployment scripts (user provides OmniParser URL)
- ❌ Elaborate semantic tags (just store text)

## 10. Timeline

**Day 1:**
- Types, Builder with temporal clustering
- Basic test with synthetic flickering data

**Day 2:**
- Locator with OCR matching
- Demo: find element across resolutions

## 11. Success Metric

Can find a text button ("Save", "Submit", etc.) across:
- Different screenshot resolutions
- 30% frame-to-frame detection dropout

That's it. Ship this, then iterate.

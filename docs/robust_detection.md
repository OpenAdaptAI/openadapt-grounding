# Robust Element Detection

**Problem:** OmniParser may fail to detect an element the user clicked on.

**Solution:** Iterative detection with image transformations until element is found.

---

## 1. Why Detection Fails

| Cause | Example | Solution |
|-------|---------|----------|
| Element too small | 16x16 icon in 4K screenshot | Crop + upscale |
| Low contrast | Gray text on gray background | Brightness/contrast adjustment |
| Cluttered context | Dense toolbar | Smaller crop isolates element |
| Center bias | Element at screen edge | Offset crops |
| Below confidence threshold | Faint text | Multiple attempts with augmentation |

## 2. Detection Strategy

```
robust_detect(image, click_xy, max_depth=4):
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Parse full image                 │
│    → Element found at click?        │──Yes──▶ Return element
└─────────────────────────────────────┘
    │ No
    ▼
┌─────────────────────────────────────┐
│ 2. Crop around click (400x400)      │
│    → Parse crop                     │──Yes──▶ Return element
│    → Element found?                 │
└─────────────────────────────────────┘
    │ No
    ▼
┌─────────────────────────────────────┐
│ 3. Try offset crops (3x3 grid)      │
│    → Parse each                     │──Yes──▶ Return element
│    → Element found in any?          │
└─────────────────────────────────────┘
    │ No
    ▼
┌─────────────────────────────────────┐
│ 4. Smaller crop (200x200) + 2x      │
│    upscale → Parse                  │──Yes──▶ Return element
└─────────────────────────────────────┘
    │ No
    ▼
┌─────────────────────────────────────┐
│ 5. Image augmentations:             │
│    - Brightness ±20%                │
│    - Contrast 1.5x                  │──Yes──▶ Return element
│    - Grayscale + sharpen            │
│    → Parse each variation           │
└─────────────────────────────────────┘
    │ No
    ▼
┌─────────────────────────────────────┐
│ 6. Super-resolution upscale         │
│    → Parse                          │──Yes──▶ Return element
└─────────────────────────────────────┘
    │ No
    ▼
┌─────────────────────────────────────┐
│ 7. Fallback: Local OCR only         │
│    → Return text at click location  │
└─────────────────────────────────────┘
```

## 3. Transformation Parameters

### 3.1 Crop Sizes
```python
CROP_SIZES = [400, 200, 100]  # pixels, centered on click
```

### 3.2 Offset Grid
```python
OFFSETS = [
    (-50, -50), (0, -50), (50, -50),
    (-50,   0), (0,   0), (50,   0),
    (-50,  50), (0,  50), (50,  50),
]
```

### 3.3 Image Augmentations
```python
AUGMENTATIONS = [
    {"brightness": 1.2},           # +20%
    {"brightness": 0.8},           # -20%
    {"contrast": 1.5},
    {"sharpen": True},
    {"grayscale": True},
    {"grayscale": True, "sharpen": True},
    {"equalize_histogram": True},  # Auto contrast
]
```

### 3.4 Scale Factors
```python
UPSCALE_FACTORS = [1.5, 2.0, 3.0]
```

## 4. Implementation

```python
from dataclasses import dataclass
from typing import Optional, List, Tuple
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import itertools

@dataclass
class DetectionResult:
    found: bool
    element: Optional[dict] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    attempts: int = 0
    successful_transform: Optional[str] = None

class RobustDetector:
    def __init__(self, parser_client, max_attempts: int = 50):
        self.parser = parser_client
        self.max_attempts = max_attempts

    def detect(
        self,
        image: Image.Image,
        click_xy: Tuple[int, int]
    ) -> DetectionResult:
        """Attempt to detect element at click location."""
        attempts = 0

        # Strategy 1: Full image
        attempts += 1
        element = self._find_at_click(image, click_xy)
        if element:
            return DetectionResult(True, element, attempts=attempts,
                                   successful_transform="original")

        # Strategy 2-6: Crops and augmentations
        for transform_name, transformed, adjusted_click in self._generate_variations(image, click_xy):
            if attempts >= self.max_attempts:
                break
            attempts += 1
            element = self._find_at_click(transformed, adjusted_click)
            if element:
                return DetectionResult(True, element, attempts=attempts,
                                       successful_transform=transform_name)

        return DetectionResult(False, attempts=attempts)

    def _generate_variations(self, image, click_xy):
        """Generate image variations to try."""
        cx, cy = click_xy
        w, h = image.size

        # Centered crops at different sizes
        for size in [400, 200, 100]:
            crop, adj_click = self._crop_around(image, click_xy, size)
            if crop:
                yield f"crop_{size}", crop, adj_click

                # Upscaled version
                upscaled = crop.resize((size*2, size*2), Image.Resampling.LANCZOS)
                yield f"crop_{size}_2x", upscaled, (adj_click[0]*2, adj_click[1]*2)

        # Offset crops
        for ox, oy in [(-50, -50), (50, -50), (-50, 50), (50, 50)]:
            crop, adj_click = self._crop_around(image, (cx+ox, cy+oy), 300)
            if crop:
                yield f"crop_offset_{ox}_{oy}", crop, adj_click

        # Augmentations on best crop size
        crop, adj_click = self._crop_around(image, click_xy, 300)
        if crop:
            # Brightness variations
            for factor in [0.8, 1.2]:
                aug = ImageEnhance.Brightness(crop).enhance(factor)
                yield f"brightness_{factor}", aug, adj_click

            # Contrast
            aug = ImageEnhance.Contrast(crop).enhance(1.5)
            yield "contrast_1.5", aug, adj_click

            # Sharpen
            aug = crop.filter(ImageFilter.SHARPEN)
            yield "sharpen", aug, adj_click

            # Grayscale
            aug = ImageOps.grayscale(crop).convert("RGB")
            yield "grayscale", aug, adj_click

            # Histogram equalization
            aug = ImageOps.equalize(crop)
            yield "equalize", aug, adj_click

    def _crop_around(self, image, center, size):
        """Crop image around center point."""
        cx, cy = center
        w, h = image.size
        half = size // 2

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        if x2 - x1 < 50 or y2 - y1 < 50:
            return None, None

        crop = image.crop((x1, y1, x2, y2))
        adj_click = (cx - x1, cy - y1)
        return crop, adj_click

    def _find_at_click(self, image, click_xy):
        """Check if any detected element contains the click point."""
        elements = self.parser.parse(image)
        cx, cy = click_xy
        w, h = image.size

        # Normalize click to 0-1
        ncx, ncy = cx / w, cy / h

        for elem in elements:
            bbox = elem.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                # Add small padding
                pad = 0.02
                if (x1 - pad <= ncx <= x2 + pad and
                    y1 - pad <= ncy <= y2 + pad):
                    return elem
        return None
```

## 5. Optimization

### 5.1 Early Termination
Stop as soon as any variation succeeds.

### 5.2 Parallel Execution
Run multiple variations in parallel (batch to GPU).

### 5.3 Caching
Cache OmniParser results for identical image hashes.

### 5.4 Priority Ordering
Order transformations by historical success rate:
```python
# Track which transforms work most often
transform_success_rate = {
    "crop_200_2x": 0.45,      # Most effective
    "crop_300": 0.30,
    "brightness_1.2": 0.15,
    # ...
}
# Try high-success transforms first
```

## 6. Metrics

| Metric | Description |
|--------|-------------|
| **Detection Rate** | % of clicks where element found |
| **Attempts to Success** | Average iterations needed |
| **Transform Effectiveness** | Success rate per transform type |
| **Latency** | Time from click to detection |
| **False Positive Rate** | Wrong element detected |

---

## 7. Next Steps

1. Implement `RobustDetector` class
2. Build evaluation harness (see `docs/evaluation.md`)
3. Curate test dataset
4. Measure baseline vs robust detection
5. Tune transformation parameters based on results

# Evaluation Harness

**Goal:** Measure how well robust detection finds elements at click locations.

> **Related Documents:**
> - [Literature Review](./literature_review.md) - SOTA analysis of UI grounding methods
> - [Experiment Plan](./experiment_plan.md) - Detailed comparison of OmniParser vs UI-TARS

---

## Overview

Based on our [literature review](./literature_review.md), we are comparing multiple approaches:

| Method | Expected Accuracy | Notes |
|--------|-------------------|-------|
| OmniParser (baseline) | ~40% | Current approach |
| OmniParser + ScreenSeekeR | ~55% | LLM-guided cropping |
| UI-TARS 1.5 | ~62% | SOTA model |
| UI-TARS + ScreenSeekeR | ~70%+ | Combined approach |

See the [experiment plan](./experiment_plan.md) for full methodology.

---

## 1. What We're Measuring

| Question | Metric |
|----------|--------|
| Can we find the clicked element? | **Detection Rate** |
| How accurate is the bounding box? | **IoU** (Intersection over Union) |
| How many attempts needed? | **Attempts to Success** |
| How long does it take? | **Latency (ms)** |
| Do we find the wrong element? | **False Positive Rate** |

## 2. Ground Truth Requirements

For each test sample, we need:
- Screenshot image
- Click location (x, y in pixels)
- Ground truth bounding box of clicked element
- Element metadata (text, type)

## 3. Dataset Format

### 3.1 Directory Structure
```
evaluation/
├── datasets/
│   ├── synthetic/           # Generated UIs (automatic ground truth)
│   │   ├── samples/
│   │   │   ├── 001.png
│   │   │   ├── 002.png
│   │   │   └── ...
│   │   └── annotations.json
│   │
│   ├── curated/             # Real screenshots (manual annotation)
│   │   ├── samples/
│   │   └── annotations.json
│   │
│   └── recorded/            # From real user sessions
│       ├── samples/
│       └── annotations.json
│
├── results/                 # Evaluation outputs
│   ├── baseline/
│   └── robust/
│
└── harness.py               # Evaluation runner
```

### 3.2 Annotation Schema
```json
{
  "version": "1.0",
  "dataset": "curated",
  "samples": [
    {
      "id": "sample_001",
      "image": "samples/001.png",
      "width": 1920,
      "height": 1080,
      "elements": [
        {
          "id": "elem_001",
          "bbox": [0.10, 0.20, 0.25, 0.28],
          "text": "Login",
          "type": "button",
          "click_point": [0.175, 0.24]
        },
        {
          "id": "elem_002",
          "bbox": [0.10, 0.32, 0.25, 0.40],
          "text": "Sign Up",
          "type": "button",
          "click_point": [0.175, 0.36]
        }
      ]
    }
  ]
}
```

**Note:** All coordinates are normalized (0-1).

## 4. Dataset Curation

### 4.1 Synthetic Dataset (Automatic)
Generate fake UIs with known ground truth:

```python
def generate_synthetic_sample(seed: int) -> dict:
    """Generate a synthetic UI with random buttons/text."""
    img = Image.new('RGB', (800, 600), '#f0f0f0')
    draw = ImageDraw.Draw(img)
    elements = []

    # Random buttons
    for i in range(random.randint(3, 8)):
        x = random.randint(50, 600)
        y = random.randint(50, 500)
        w = random.randint(80, 150)
        h = random.randint(30, 50)
        text = random.choice(["Submit", "Cancel", "OK", "Save", "Delete", ...])

        draw.rectangle([x, y, x+w, y+h], fill=random_color())
        draw.text((x+10, y+10), text, fill='white')

        elements.append({
            "bbox": [x/800, y/600, (x+w)/800, (y+h)/600],
            "text": text,
            "type": "button",
            "click_point": [(x+w/2)/800, (y+h/2)/600]
        })

    return {"image": img, "elements": elements}
```

**Pros:** Unlimited samples, perfect ground truth
**Cons:** May not reflect real-world complexity

### 4.2 Semi-Automated Curation (Recommended)

1. **Collect screenshots** from real apps
2. **Run OmniParser** to get initial detections
3. **Human review** in annotation tool:
   - Accept correct detections
   - Adjust incorrect bounding boxes
   - Add missed elements
   - Mark representative click points
4. **Export** to annotation format

### 4.3 Annotation Tool (CLI)

Simple terminal-based annotation:

```python
def annotate_sample(image_path: str) -> dict:
    """Interactive CLI annotation."""
    img = Image.open(image_path)
    elements = omniparser.parse(img)

    print(f"OmniParser found {len(elements)} elements")
    print("For each element, enter: [a]ccept, [r]eject, [e]dit, [s]kip")

    accepted = []
    for i, elem in enumerate(elements):
        show_element(img, elem)  # Display with bbox highlighted
        choice = input(f"Element {i+1} '{elem.get('content', '')[:20]}': ")

        if choice == 'a':
            click = get_click_point(elem)  # Center of bbox
            accepted.append({**elem, "click_point": click})
        elif choice == 'e':
            edited = edit_element(elem)  # Manual bbox adjustment
            accepted.append(edited)

    # Add missed elements
    while input("Add missed element? [y/n]: ") == 'y':
        elem = manual_annotate(img)
        accepted.append(elem)

    return {"image": image_path, "elements": accepted}
```

### 4.4 Click Recording (Most Realistic)

Record real user interactions:

```python
def record_session():
    """Record clicks with screenshots."""
    samples = []
    while True:
        # Wait for click
        click_xy = wait_for_click()
        screenshot = capture_screenshot()

        # Human labels what they clicked
        label = input("What did you click? (text/description): ")

        samples.append({
            "image": save_screenshot(screenshot),
            "click_xy": click_xy,
            "label": label,
            # Ground truth bbox determined by human later
        })
```

## 5. Evaluation Harness

```python
from dataclasses import dataclass
from typing import List
import json
import time

@dataclass
class EvalResult:
    sample_id: str
    element_id: str
    detected: bool
    iou: float
    attempts: int
    latency_ms: float
    transform_used: str

def evaluate_dataset(
    detector,
    dataset_path: str,
    output_path: str
) -> dict:
    """Run evaluation on a dataset."""
    with open(dataset_path) as f:
        dataset = json.load(f)

    results = []
    for sample in dataset["samples"]:
        img = Image.open(sample["image"])
        w, h = img.size

        for elem in sample["elements"]:
            # Convert normalized click to pixels
            cx = int(elem["click_point"][0] * w)
            cy = int(elem["click_point"][1] * h)

            # Run detection
            start = time.time()
            result = detector.detect(img, (cx, cy))
            latency = (time.time() - start) * 1000

            # Calculate IoU if detected
            iou = 0.0
            if result.found and result.bbox:
                iou = calculate_iou(elem["bbox"], result.bbox)

            results.append(EvalResult(
                sample_id=sample["id"],
                element_id=elem["id"],
                detected=result.found,
                iou=iou,
                attempts=result.attempts,
                latency_ms=latency,
                transform_used=result.successful_transform or "none"
            ))

    # Aggregate metrics
    metrics = compute_metrics(results)

    # Save results
    save_results(results, metrics, output_path)

    return metrics

def compute_metrics(results: List[EvalResult]) -> dict:
    """Compute aggregate metrics."""
    total = len(results)
    detected = sum(1 for r in results if r.detected)

    return {
        "detection_rate": detected / total if total > 0 else 0,
        "mean_iou": mean([r.iou for r in results if r.detected]),
        "mean_attempts": mean([r.attempts for r in results]),
        "mean_latency_ms": mean([r.latency_ms for r in results]),
        "transform_breakdown": count_transforms(results),
    }

def calculate_iou(bbox1, bbox2) -> float:
    """Calculate Intersection over Union."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
```

## 6. Running Evaluation

```bash
# Generate synthetic dataset
uv run python -m openadapt_grounding.eval generate --type synthetic --count 500

# Run OmniParser baseline
uv run python -m openadapt_grounding.eval run --method omniparser --dataset synthetic

# Run OmniParser with ScreenSeekeR cropping
uv run python -m openadapt_grounding.eval run --method omniparser-screenseeker --dataset synthetic

# Run UI-TARS baseline
uv run python -m openadapt_grounding.eval run --method uitars --dataset synthetic

# Compare all methods
uv run python -m openadapt_grounding.eval compare --output results/comparison.json
```

## 7. Expected Output

```
============================================================
Evaluation Results: OmniParser vs UI-TARS
============================================================

Dataset: synthetic (500 samples)

Method                      Accuracy    Latency
─────────────────────────────────────────────────────
OmniParser (baseline)       ~40%        250ms
OmniParser + ScreenSeekeR   ~55%        1200ms
UI-TARS 1.5 (baseline)      ~62%        350ms
UI-TARS + ScreenSeekeR      ~70%+       1500ms

By Element Size:
  Size          OmniParser    UI-TARS
  <32px         ~15%          ~35%
  32-100px      ~45%          ~65%
  >100px        ~70%          ~85%

Failure Analysis:
  - Small icons (<32px) remain hardest
  - Text elements detected more reliably than icons
  - ScreenSeekeR cropping helps most on small elements
```

## 8. Curation Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    CURATION WORKFLOW                        │
└─────────────────────────────────────────────────────────────┘

1. COLLECT
   ├── Take screenshots of target applications
   ├── Include variety: light/dark themes, resolutions, apps
   └── Aim for 50-100 screenshots initially

2. AUTO-ANNOTATE
   ├── Run OmniParser on each screenshot
   └── Generate initial annotations.json

3. HUMAN REVIEW
   ├── For each detected element:
   │   ├── Accept (bbox correct, element meaningful)
   │   ├── Adjust (fix bbox boundaries)
   │   └── Reject (false positive)
   ├── Add missed elements manually
   └── Mark click points (typically bbox center)

4. VALIDATE
   ├── Run evaluation with known-good detector
   ├── Check for annotation errors
   └── Fix outliers

5. VERSION & SHARE
   ├── Commit dataset to repo
   ├── Document collection methodology
   └── Track changes over time
```

## 9. Next Steps

See [experiment_plan.md](./experiment_plan.md) for detailed implementation plan.

### Phase 1: Infrastructure
- [ ] Deploy UI-TARS 1.5 (GPU server)
- [ ] Implement unified evaluation harness
- [ ] Create synthetic dataset generator (500 samples)
- [ ] Download ScreenSpot subset (200 samples)

### Phase 2: Baseline Evaluation
- [ ] Run OmniParser baseline on all datasets
- [ ] Run UI-TARS baseline on all datasets
- [ ] Generate initial comparison plots
- [ ] Identify failure cases

### Phase 3: Advanced Methods
- [ ] Implement ScreenSeekeR-style cropping for both models
- [ ] Run full evaluation matrix
- [ ] Compare all 6 methods (2 models × 3 cropping strategies)

### Phase 4: Analysis
- [ ] Generate final plots (bar charts, scatter, line, confusion matrix)
- [ ] Analyze failure cases by element size/type
- [ ] Write recommendations for production
- [ ] Document findings

### Key Metrics to Report
- Detection rate by dataset (synthetic, ScreenSpot, real)
- Detection rate by element size (<32px, 32-100px, >100px)
- Latency vs accuracy tradeoff
- Cropping strategy effectiveness

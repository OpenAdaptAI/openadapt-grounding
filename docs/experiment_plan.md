# Experiment Plan: Grounding Model Comparison

## Objective

Compare UI element grounding approaches to determine optimal strategy for click-to-element detection:

1. **OmniParser** (baseline) - Current approach
2. **OmniParser + ScreenSeekeR-style cropping** - Progressive cropping enhancement
3. **UI-TARS 1.5** - SOTA model
4. **UI-TARS + ScreenSeekeR-style cropping** - Combined approach

---

## Hypotheses

| ID | Hypothesis | Expected Result |
|----|------------|-----------------|
| H1 | UI-TARS outperforms OmniParser on both synthetic and real data | UI-TARS +20-30% accuracy |
| H2 | Progressive cropping improves both models | +50-100% relative improvement |
| H3 | Combined approach (UI-TARS + cropping) achieves best results | >70% accuracy |
| H4 | Latency increases with cropping but remains acceptable | <2s per detection |

---

## Experimental Design

### Independent Variables

| Variable | Levels |
|----------|--------|
| Base Model | OmniParser, UI-TARS 1.5 |
| Cropping Strategy | None, Fixed (our current), ScreenSeekeR-style |
| Dataset | Synthetic, ScreenSpot, Real (curated) |

### Dependent Variables

| Metric | Description | Target |
|--------|-------------|--------|
| Detection Rate | % of elements correctly located | >80% |
| IoU | Bounding box accuracy | >0.7 |
| Latency | Time per detection (ms) | <2000 |
| Attempts | Number of crops/transforms tried | <5 |

### Control Variables

- Hardware: Same GPU (e.g., A100 or 4090)
- Image resolution: Standardized to 1920x1080
- Confidence threshold: 0.5 for all models

---

## Datasets

### 1. Synthetic Dataset (N=500)

Auto-generated UIs with perfect ground truth.

```
synthetic/
├── simple/          # Large buttons, high contrast (N=100)
├── medium/          # Standard UI complexity (N=200)
├── hard/            # Small icons, dense layouts (N=100)
└── professional/    # Mimics ScreenSpot-Pro apps (N=100)
```

**Generation parameters:**
- Element sizes: 16px-200px
- Element types: buttons, icons, text fields, checkboxes
- Backgrounds: light, dark, gradients
- Density: 5-50 elements per screen

### 2. ScreenSpot Subset (N=200)

Sampled from public ScreenSpot benchmark for comparability.

```
screenspot/
├── mobile/          # iOS, Android (N=60)
├── desktop/         # macOS, Windows (N=80)
└── web/             # Various websites (N=60)
```

### 3. Real Curated Dataset (N=100)

Screenshots from actual usage, manually annotated.

```
real/
├── productivity/    # VSCode, Slack, Chrome (N=40)
├── creative/        # Figma, Photoshop (N=30)
└── system/          # OS dialogs, settings (N=30)
```

---

## Methods

### Method 1: Baseline (OmniParser)

```python
def baseline_omniparser(image, click_xy):
    """Single-pass OmniParser detection."""
    elements = omniparser.parse(image)
    return find_element_at_point(elements, click_xy)
```

### Method 2: OmniParser + Fixed Cropping (Current)

```python
def omniparser_fixed_crop(image, click_xy):
    """Our current transform-based approach."""
    transforms = [
        lambda img: img,  # Original
        lambda img: crop_around(img, click_xy, 200),
        lambda img: crop_around(img, click_xy, 300),
        lambda img: adjust_brightness(img, 1.2),
        lambda img: to_grayscale(img),
    ]
    for transform in transforms:
        elements = omniparser.parse(transform(image))
        if element := find_element_at_point(elements, click_xy):
            return element
    return None
```

### Method 3: OmniParser + ScreenSeekeR-style Cropping

```python
def omniparser_screenSeekeR(image, click_xy, target_description=None):
    """LLM-guided progressive cropping."""
    # Step 1: Get region predictions from LLM
    regions = llm.predict_ui_regions(image, click_xy, target_description)
    # e.g., ["toolbar", "menu bar", "left sidebar"]

    candidates = []
    for region in regions:
        # Step 2: Crop to predicted region
        cropped = crop_to_region(image, region)

        # Step 3: Run detection on cropped region
        elements = omniparser.parse(cropped)

        # Step 4: Map back to original coordinates
        for elem in elements:
            elem.bbox = map_to_original(elem.bbox, region)
            candidates.append(elem)

    # Step 5: Vote and select best candidate
    return gaussian_vote(candidates, click_xy)
```

### Method 4: UI-TARS Baseline

```python
def baseline_uitars(image, click_xy):
    """Single-pass UI-TARS detection."""
    # UI-TARS uses different API - grounding query
    prompt = f"Find the UI element at coordinates ({click_xy[0]}, {click_xy[1]})"
    result = uitars.ground(image, prompt)
    return result
```

### Method 5: UI-TARS + ScreenSeekeR-style Cropping

```python
def uitars_screenSeekeR(image, click_xy, target_description=None):
    """UI-TARS with progressive cropping."""
    # Same cropping strategy as Method 3
    regions = llm.predict_ui_regions(image, click_xy, target_description)

    candidates = []
    for region in regions:
        cropped = crop_to_region(image, region)
        result = uitars.ground(cropped, prompt)
        if result:
            result.bbox = map_to_original(result.bbox, region)
            candidates.append(result)

    return gaussian_vote(candidates, click_xy)
```

---

## Implementation Plan

### Phase 1: Infrastructure (Week 1)

- [ ] Set up UI-TARS 1.5 deployment (GPU server)
- [ ] Implement unified evaluation harness
- [ ] Create synthetic dataset generator
- [ ] Download/prepare ScreenSpot subset

### Phase 2: Baseline Evaluation (Week 2)

- [ ] Run OmniParser baseline on all datasets
- [ ] Run UI-TARS baseline on all datasets
- [ ] Compute metrics and generate initial plots
- [ ] Identify failure cases

### Phase 3: Cropping Strategies (Week 3)

- [ ] Implement fixed cropping for both models
- [ ] Implement ScreenSeekeR-style cropping
- [ ] Run evaluations on all datasets
- [ ] Compare cropping strategies

### Phase 4: Analysis & Reporting (Week 4)

- [ ] Generate final comparison plots
- [ ] Analyze failure cases
- [ ] Write up findings
- [ ] Recommend production approach

---

## Evaluation Metrics

### Primary Metrics

```python
@dataclass
class EvalMetrics:
    detection_rate: float      # % found within IoU threshold
    mean_iou: float           # Average IoU for detected elements
    mean_latency_ms: float    # Average detection time
    mean_attempts: float      # Average crops/transforms tried
```

### Secondary Metrics

```python
@dataclass
class DetailedMetrics:
    # Breakdown by element size
    detection_rate_small: float   # <32px
    detection_rate_medium: float  # 32-100px
    detection_rate_large: float   # >100px

    # Breakdown by element type
    detection_rate_text: float
    detection_rate_icon: float
    detection_rate_button: float

    # Failure analysis
    false_positive_rate: float
    wrong_element_rate: float     # Detected but wrong element
```

---

## Visualization Plan

### Plot 1: Overall Comparison (Bar Chart)

```
Detection Rate by Method
========================
                    Synthetic  ScreenSpot  Real
OmniParser          [===]      [==]        [==]
OmniParser+Crop     [====]     [===]       [===]
OmniParser+SeekeR   [=====]    [====]      [====]
UI-TARS             [======]   [=====]     [=====]
UI-TARS+Crop        [======]   [=====]     [=====]
UI-TARS+SeekeR      [=======]  [======]    [======]
```

### Plot 2: Accuracy vs Latency (Scatter)

```
Accuracy vs Latency Tradeoff
============================
     |           * UI-TARS+SeekeR
  A  |       * UI-TARS
  c  |     * OmniParser+SeekeR
  c  |   * OmniParser+Crop
     | * OmniParser
     +------------------------
           Latency (ms)
```

### Plot 3: Performance by Element Size (Line)

```
Detection Rate by Element Size
==============================
100%|----*----*----*----*
    |   /    /    /    /
 50%|--/----/----/----/
    | /    /    /    /
  0%|/----/----/----/----
    16   32   64  128  256
         Element Size (px)

— OmniParser  — UI-TARS  — +SeekeR
```

### Plot 4: Cumulative Detection (Line)

```
Detection Rate vs Attempts
==========================
100%|            *---*---*
    |        *--/
 80%|    *--/
    |  */
 60%|*/
    |
    +---+---+---+---+---
    1   2   3   4   5
        Attempts

Shows how detection rate improves with more cropping attempts
```

### Plot 5: Confusion Matrix (Heatmap)

```
Element Type Confusion
======================
              Predicted
            Btn  Icon  Text  None
Actual Btn  [85] [ 5]  [ 2]  [ 8]
      Icon  [ 3] [72]  [ 5]  [20]
      Text  [ 1] [ 2]  [90]  [ 7]
```

### Plot 6: Failure Case Gallery

Visual grid showing:
- Screenshot with ground truth bbox (green)
- Detected bbox (red/yellow)
- Click point (blue dot)
- Method used
- IoU score

---

## Success Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Best method detection rate | >80% | Usable for production |
| Latency | <2000ms | Acceptable UX |
| Improvement over baseline | >30% relative | Worth the complexity |
| Consistent across datasets | <10% variance | Generalizable |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| UI-TARS deployment issues | High | Fall back to API/cloud |
| LLM costs for ScreenSeekeR | Medium | Cache predictions, batch |
| Synthetic data not representative | Medium | Validate on real data |
| GPU availability | Medium | Use cloud instances |

---

## Deliverables

1. **Code**
   - `evaluation/harness.py` - Unified evaluation framework
   - `evaluation/datasets/` - Dataset generators and loaders
   - `evaluation/methods/` - All detection methods
   - `evaluation/plots.py` - Visualization generation

2. **Data**
   - Synthetic dataset (500 samples)
   - ScreenSpot subset (200 samples)
   - Real curated dataset (100 samples)
   - All evaluation results (JSON)

3. **Documentation**
   - This experiment plan
   - Results summary with plots
   - Recommendations for production

4. **Artifacts**
   - Trained/deployed models
   - Evaluation plots (PNG/PDF)
   - Interactive results viewer (optional)

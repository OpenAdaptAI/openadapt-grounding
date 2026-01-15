# Evaluation Framework Implementation

This document describes the implementation of the evaluation framework for comparing UI grounding methods.

## Overview

The evaluation framework enables systematic comparison of 6 grounding methods across synthetic and real datasets:

| Method | Description |
|--------|-------------|
| `omniparser` | OmniParser baseline (full image) |
| `omniparser-fixed` | OmniParser + fixed cropping (200, 300, 500px) |
| `omniparser-screenseeker` | OmniParser + ScreenSeekeR-style heuristic cropping |
| `uitars` | UI-TARS baseline (full image) |
| `uitars-fixed` | UI-TARS + fixed cropping |
| `uitars-screenseeker` | UI-TARS + ScreenSeekeR-style cropping |

## Module Structure

```
src/openadapt_grounding/eval/
├── __init__.py              # Public API
├── __main__.py              # CLI entry point
├── cli.py                   # CLI implementation
├── config.py                # Settings
├── dataset/
│   ├── schema.py            # Dataset, Sample, AnnotatedElement
│   └── synthetic.py         # Synthetic generator
├── methods/
│   ├── base.py              # EvaluationMethod interface
│   ├── omniparser.py        # OmniParser method
│   ├── uitars.py            # UI-TARS method
│   └── cropping.py          # Cropping strategies
├── metrics/
│   ├── types.py             # ElementResult, MethodMetrics
│   └── compute.py           # IoU, aggregation
├── results/
│   ├── storage.py           # Save/load JSON
│   └── compare.py           # Multi-method comparison
└── visualization/
    ├── charts.py            # matplotlib charts
    └── tables.py            # Console/markdown tables
```

## Dataset Schema

### AnnotatedElement

```python
@dataclass
class AnnotatedElement:
    id: str
    bbox: Tuple[float, float, float, float]  # (x, y, w, h) normalized 0-1
    text: Optional[str]
    element_type: ElementType  # button, icon, text, etc.
    instruction: Optional[str]  # For UI-TARS grounding
```

### Sample

```python
@dataclass
class Sample:
    id: str
    image_path: str  # Relative to dataset root
    width: int
    height: int
    elements: List[AnnotatedElement]
    metadata: Dict[str, Any]
```

### Dataset JSON Format

```json
{
  "name": "synthetic",
  "version": "1.0",
  "metadata": {"num_samples": 500},
  "samples": [
    {
      "id": "synthetic_0001",
      "image_path": "samples/synthetic_0001.png",
      "width": 1920,
      "height": 1080,
      "elements": [
        {
          "id": "elem_001",
          "bbox": [0.1, 0.2, 0.15, 0.05],
          "text": "Login",
          "element_type": "button",
          "instruction": "Click on the 'Login' button"
        }
      ]
    }
  ]
}
```

## Evaluation Methods

### EvaluationMethod Interface

```python
class EvaluationMethod(ABC):
    @property
    def name(self) -> str: ...

    def evaluate_element(
        self, image: Image, target: AnnotatedElement
    ) -> EvaluationPrediction: ...

    def is_available(self) -> bool: ...
```

### OmniParser Method

1. Parse full image (or cropped regions)
2. Find element with best match to target (text + IoU)
3. Return matched element's center as click point

**Matching Score:**
- IoU between detected and target bbox
- Text similarity bonus (+0.3 for exact match, +0.15 for partial)
- Distance penalty for far matches

### UI-TARS Method

1. Build instruction from target element text/type
2. Call `client.ground(image, instruction)`
3. Check if returned point falls within target bbox (with tolerance)

**Instruction Generation:**
```python
if element.instruction:
    return element.instruction
elif element.text:
    return f"Click on the '{element.text}' {element.element_type}"
else:
    return f"Click on the {element.element_type}"
```

## Cropping Strategies

### NoCropping (baseline)
- Single full-image evaluation

### FixedCropping
- Full image + crops at 200px, 300px, 500px centered on target
- Returns first successful match

### ScreenSeekeRCropping
- Full image + heuristic UI regions:
  - Top toolbar (y < 0.15)
  - Left sidebar (x < 0.25)
  - Right panel (x > 0.75)
  - Center content
  - Quadrant containing target

## Metrics

### Per-Element Results

```python
@dataclass
class ElementResult:
    sample_id: str
    element_id: str
    found: bool
    iou: float
    latency_ms: float
    attempts: int
    size_category: str  # small, medium, large
    element_type: str
    distance_from_target: float
```

### Aggregated Metrics

```python
@dataclass
class MethodMetrics:
    method_name: str
    dataset_name: str
    detection_rate: float      # % elements found
    mean_iou: float           # Average IoU (when bbox available)
    mean_latency_ms: float    # Average time per element
    mean_attempts: float      # Average crop attempts
    detection_rate_small: float   # <32px elements
    detection_rate_medium: float  # 32-100px
    detection_rate_large: float   # >100px
    detection_rate_by_type: Dict[str, float]
```

## Synthetic Dataset Generator

### Element Types
- **Buttons**: Colored rectangles with text
- **Links**: Underlined text
- **Icons**: Circular placeholders with initials
- **Text**: Plain text labels
- **Text Fields**: Bordered rectangles with placeholder

### Difficulty Levels
| Level | Elements | Min Size | Max Size |
|-------|----------|----------|----------|
| easy | 3-8 | 80px | 200px |
| medium | 8-20 | 40px | 150px |
| hard | 20-50 | 16px | 80px |

### Distribution
Default: 15% easy-light, 5% easy-dark, 30% medium-light, 15% medium-dark, 20% hard-light, 15% hard-dark

## CLI Usage

```bash
# Generate dataset
python -m openadapt_grounding.eval generate --type synthetic --count 500

# Run evaluation
python -m openadapt_grounding.eval run --method omniparser --dataset synthetic

# Compare results
python -m openadapt_grounding.eval compare --charts-dir evaluation/charts

# List methods
python -m openadapt_grounding.eval list
```

## Visualization

### Charts Generated
1. **Detection Rate Bar Chart**: Overall rate by method
2. **Size Breakdown Chart**: Rate by element size (small/medium/large)
3. **Latency vs Accuracy**: Scatter plot showing tradeoffs

### Console Table
```
================================================================================
EVALUATION RESULTS
================================================================================
Method                      Dataset      Det.Rate  IoU      Latency   Attempts
--------------------------------------------------------------------------------
OmniParser + baseline       synthetic    65.2%     0.452    1250ms    1.0
UI-TARS + baseline          synthetic    78.4%     0.000    890ms     1.0
================================================================================
```

## Configuration

Environment variables (prefix `EVAL_`):
- `EVAL_OMNIPARSER_URL`: OmniParser server URL
- `EVAL_UITARS_URL`: UI-TARS server URL
- `EVAL_DATASETS_DIR`: Dataset directory
- `EVAL_RESULTS_DIR`: Results directory
- `EVAL_CHARTS_DIR`: Charts directory

## Future Enhancements

1. **ScreenSpot Dataset**: Add loader for public benchmark subset
2. **LLM Cropping**: Implement actual LLM-guided region prediction
3. **Confidence Thresholds**: Add configurable thresholds
4. **Failure Gallery**: Visualize failure cases with screenshots
5. **Real Dataset Curation**: Tool for manual annotation

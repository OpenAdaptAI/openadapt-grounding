# OpenAdapt Grounding

**Robust UI element localization for automation.**

Turn flakey single-frame detections into stable, reliable element coordinates.

## The Problem

Vision models like OmniParser miss elements randomly frame-to-frame ("flickering"). Template matching breaks with resolution/theme changes.

![Raw Flickering Detection](assets/raw_flickering.gif)

*Left: Raw detections showing frame-to-frame flickering*

## The Solution

1. **Temporal Smoothing**: Aggregate detections across frames, keep only stable elements
2. **Text Anchoring**: Match elements by OCR text (resolution-independent)

![Side-by-Side Comparison](assets/comparison.gif)

*Side-by-side: Raw flickering (left) vs Stabilized detection (right)*

## Results

### Detection Stability

| Metric | Raw (30% dropout) | Stabilized |
|--------|-------------------|------------|
| Avg Detection Rate | ~60-70% | **80-100%** |
| Min Detection Rate | ~40% | **Consistent** |
| Consistency | Flickering | **Stable** |

### Resolution Robustness

| Scale | Resolution | Elements Found | Status |
|-------|------------|----------------|--------|
| 1.0x | 800x600 | All | ✓ |
| 1.25x | 1000x750 | All | ✓ |
| 1.5x | 1200x900 | All | ✓ |
| 2.0x | 1600x1200 | All | ✓ |

### Visual Output

![Stabilized Detection](assets/stable_detection.png)

*Stable elements after temporal filtering*

## Quick Start

```bash
uv pip install openadapt-grounding
```

### Build a Registry (Offline)

```python
from openadapt_grounding import RegistryBuilder, Element

# Add detections from multiple frames
builder = RegistryBuilder()
builder.add_frame([
    Element(bounds=(0.3, 0.2, 0.2, 0.05), text="Login"),
    Element(bounds=(0.3, 0.3, 0.2, 0.05), text="Cancel"),
])
# ... add more frames

# Build registry (keeps elements in >50% of frames)
registry = builder.build(min_stability=0.5)
registry.save("elements.json")
```

### Locate Elements (Runtime)

```python
from openadapt_grounding import ElementLocator
from PIL import Image

locator = ElementLocator("elements.json")
screenshot = Image.open("current_screen.png")

result = locator.find("Login", screenshot)
if result.found:
    # Normalized coordinates (0-1)
    print(f"Found at ({result.x:.2f}, {result.y:.2f})")

    # Convert to pixels
    px, py = result.to_pixels(width=1920, height=1080)
    print(f"Click at ({px}, {py})")
```

## Run Demo

```bash
uv run python -m openadapt_grounding.demo
```

Output:
```
============================================================
OpenAdapt Grounding Demo Results
============================================================

Registry: 5 stable elements

📊 Detection Stability:
  Raw (with 30% dropout):    70%
  Stabilized (filtered):     100%
  Improvement:               +30%

📐 Resolution Robustness:
  ✓ 1.0x (800x600): 5 elements
  ✓ 1.25x (1000x750): 5 elements
  ✓ 1.5x (1200x900): 5 elements
  ✓ 2.0x (1600x1200): 5 elements

📁 Outputs saved to: demo_output/
```

## How It Works

### Temporal Clustering

```
Frame 1: [Login ✓] [Cancel ✓] [Password ✗]  → 2/3 detected
Frame 2: [Login ✓] [Cancel ✗] [Password ✓]  → 2/3 detected
Frame 3: [Login ✓] [Cancel ✓] [Password ✓]  → 3/3 detected
...
After 10 frames:
  - "Login" seen 9/10 times → KEEP (90% stability)
  - "Cancel" seen 7/10 times → KEEP (70% stability)
  - "Password" seen 8/10 times → KEEP (80% stability)
```

### Text-Based Matching

At runtime, we use OCR to find text on screen, then match against the registry:

```python
# Registry knows "Login" button exists
# OCR finds "Login" text at (0.45, 0.35)
# → Return those coordinates with high confidence
```

## OmniParser Integration

Use with [OmniParser](https://github.com/microsoft/OmniParser) for real UI element detection:

### Deploy OmniParser Server

```bash
# Install deploy dependencies
uv pip install openadapt-grounding[deploy]

# Set AWS credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1

# Deploy to EC2 (g4dn.xlarge with T4 GPU)
python -m openadapt_grounding.deploy start

# Check status
python -m openadapt_grounding.deploy status

# Stop when done
python -m openadapt_grounding.deploy stop
```

### Use OmniParser with Temporal Smoothing

```python
from openadapt_grounding import OmniParserClient, collect_frames
from PIL import Image

# Connect to deployed server
client = OmniParserClient("http://<server-ip>:8000")

# Take a screenshot
screenshot = Image.open("screen.png")

# Run parser 10 times, keep elements in >50% of frames
registry = collect_frames(client, screenshot, num_frames=10, min_stability=0.5)
registry.save("stable_elements.json")

print(f"Found {len(registry)} stable elements")
```

### Analyze Detection Stability

```python
from openadapt_grounding import OmniParserClient, analyze_stability

client = OmniParserClient("http://<server-ip>:8000")
stats = analyze_stability(client, screenshot, num_frames=10)

print(f"Average stability: {stats['avg_stability']:.0%}")
for elem in stats['elements']:
    print(f"  {elem['text']}: {elem['stability']:.0%}")
```

## API

### `RegistryBuilder`
- `add_frame(elements)` - Add a frame's detections
- `build(min_stability=0.5)` - Build registry, filtering unstable elements

### `ElementLocator`
- `find(query, screenshot)` - Find element by text
- `find_by_uid(uid, screenshot)` - Find element by registry UID

### `LocatorResult`
- `found: bool` - Whether element was found
- `x, y: float` - Normalized coordinates (0-1)
- `confidence: float` - Match confidence
- `to_pixels(w, h)` - Convert to pixel coordinates

### `OmniParserClient`
- `is_available()` - Check if server is running
- `parse(image)` - Parse screenshot, return elements
- `parse_with_metadata(image)` - Parse with latency info

### `collect_frames(parser, image, num_frames, min_stability)`
- Run parser multiple times, build stable registry

### `analyze_stability(parser, image, num_frames)`
- Report per-element detection stability

## Development

```bash
git clone https://github.com/OpenAdaptAI/openadapt-grounding
cd openadapt-grounding
uv venv && uv pip install -e ".[dev]"
uv run pytest
```

## License

MIT

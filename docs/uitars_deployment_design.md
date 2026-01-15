# UI-TARS Deployment Design

**Goal:** Deploy UI-TARS 1.5-7B for element grounding evaluation alongside OmniParser.

---

## 1. Overview

UI-TARS 1.5-7B is ByteDance's SOTA UI grounding model achieving 61.6% on ScreenSpot-Pro (vs OmniParser's 39.6%). It's based on Qwen2.5-VL architecture and uses an OpenAI-compatible chat API.

### Key Differences from OmniParser

| Aspect | OmniParser | UI-TARS 1.5-7B |
|--------|------------|----------------|
| Architecture | Custom FastAPI server | vLLM server (OpenAI-compatible) |
| Output | Bounding boxes + text | Thought + Action (click coordinates) |
| Coordinate Format | Normalized [0-1] bbox | Absolute pixels in `<point>x y</point>` |
| Model Size | ~2GB | ~16GB (7B params) |
| VRAM Required | ~8GB | ~16GB (fp16) or ~24GB (fp32) |

---

## 2. Deployment Options

### Option A: HuggingFace Inference Endpoints (Managed)

**Pros:**
- Zero infrastructure management
- Auto-scaling
- Quick setup (~10 minutes)

**Cons:**
- ~$4/hr for L40S GPU (more expensive than self-hosted)
- Less control over configuration
- Vendor lock-in

**Setup:**
1. Go to https://endpoints.huggingface.co/catalog
2. Select `UI-TARS-1.5-7B` → Import Model
3. Configure: GPU L40S 1GPU 48G
4. Set environment: `CUDA_GRAPHS=0`, `PAYLOAD_LIMIT=8000000`
5. Set container URI: `ghcr.io/huggingface/text-generation-inference:3.2.1`

### Option B: Self-Hosted on AWS EC2 (Recommended)

**Pros:**
- ~$1/hr (g6.2xlarge with L4 24GB)
- Full control
- Consistent with OmniParser deployment
- Auto-shutdown saves costs

**Cons:**
- More setup work
- Manual scaling

---

## 3. Recommended Architecture (Option B)

```
┌─────────────────────────────────────────────────────────────┐
│                      AWS EC2 (g6.2xlarge)                   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    Docker Container                   │  │
│  │                                                       │  │
│  │   vLLM Server (OpenAI-compatible API)                │  │
│  │   └── UI-TARS-1.5-7B model                           │  │
│  │   └── Port 8000                                       │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  CloudWatch → Lambda (auto-shutdown on idle)                │
└─────────────────────────────────────────────────────────────┘
           │
           │ HTTP :8000
           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Client (Local)                          │
│                                                             │
│   UITarsClient                                              │
│   └── OpenAI SDK (POST /v1/chat/completions)               │
│   └── ui-tars action_parser (coordinate conversion)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Hardware Requirements

### EC2 Instance Options

| Instance | GPU | VRAM | vCPU | Cost/hr | Notes |
|----------|-----|------|------|---------|-------|
| g6.xlarge | 1x L4 | 24GB | 4 | $0.81 | Minimum viable |
| g6.2xlarge | 1x L4 | 24GB | 8 | $0.98 | **Recommended** |
| g5.xlarge | 1x A10G | 24GB | 4 | $1.01 | Alternative |
| g5.2xlarge | 1x A10G | 24GB | 8 | $1.21 | More CPU |

**Recommendation:** `g6.2xlarge` - L4 24GB is sufficient for 7B model with fp16, extra vCPUs help with vLLM.

### Disk: 100GB gp3 (model weights ~15GB + docker images)

---

## 5. Implementation Plan

### 5.1 Directory Structure

```
src/openadapt_grounding/
├── deploy/
│   ├── config.py           # Add UI-TARS settings
│   ├── deploy.py           # OmniParser deployment (existing)
│   ├── Dockerfile          # OmniParser (existing)
│   └── uitars/
│       ├── __init__.py
│       ├── deploy.py       # UI-TARS deployment
│       └── Dockerfile      # vLLM + UI-TARS
├── clients/
│   ├── __init__.py
│   ├── omniparser.py       # OmniParser client (existing)
│   └── uitars.py           # UI-TARS client (new)
```

### 5.2 Config Changes

Add to `config.py`:

```python
class UITarsSettings(BaseSettings):
    """UI-TARS deployment settings."""

    UITARS_PROJECT_NAME: str = "uitars"
    UITARS_MODEL_ID: str = "ByteDance-Seed/UI-TARS-1.5-7B"
    UITARS_AWS_EC2_INSTANCE_TYPE: str = "g6.2xlarge"  # L4 24GB
    UITARS_AWS_EC2_DISK_SIZE: int = 100
    UITARS_PORT: int = 8001  # Different from OmniParser
    UITARS_MAX_MODEL_LEN: int = 32768  # Reduced for memory
    UITARS_GPU_MEMORY_UTILIZATION: float = 0.90
```

### 5.3 Dockerfile (vLLM)

```dockerfile
FROM vllm/vllm-openai:v0.7.3

# Pre-download model weights
ENV HF_HOME=/root/.cache/huggingface
RUN pip install huggingface_hub && \
    huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B

# vLLM server config
ENV MODEL_ID=ByteDance-Seed/UI-TARS-1.5-7B
ENV PORT=8000

EXPOSE 8000

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "${MODEL_ID}", \
     "--host", "0.0.0.0", \
     "--port", "${PORT}", \
     "--max-model-len", "32768", \
     "--gpu-memory-utilization", "0.90", \
     "--limit-mm-per-prompt", "{\"image\":1,\"video\":0}"]
```

### 5.4 Client Implementation

```python
# src/openadapt_grounding/clients/uitars.py
import base64
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image
import io

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai not installed. Run: pip install openai")

from ui_tars.action_parser import (
    parse_action_to_structure_output,
    smart_resize,
)
from ui_tars.prompt import GROUNDING_DOUBAO


@dataclass
class GroundingResult:
    """Result of a grounding query."""
    found: bool
    x: Optional[float] = None  # Normalized 0-1
    y: Optional[float] = None  # Normalized 0-1
    confidence: float = 0.0
    raw_response: str = ""
    thought: str = ""

    def to_pixels(self, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized coords to pixel coordinates."""
        if not self.found:
            raise ValueError("Element not found")
        return int(self.x * width), int(self.y * height)


class UITarsClient:
    """Client for UI-TARS grounding API."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "not-needed",  # vLLM doesn't require auth
        model: str = "ByteDance-Seed/UI-TARS-1.5-7B",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def is_available(self) -> bool:
        """Check if server is available."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def ground(
        self,
        image: Image.Image,
        instruction: str,
        language: str = "English",
    ) -> GroundingResult:
        """Ground an element in the image by instruction.

        Args:
            image: Screenshot to search
            instruction: What to find (e.g., "Click on the Login button")
            language: Language for model output

        Returns:
            GroundingResult with normalized coordinates
        """
        # Encode image
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64_image = base64.b64encode(buffer.getvalue()).decode()

        # Build prompt
        prompt = GROUNDING_DOUBAO.format(
            language=language,
            instruction=instruction,
        )

        # Call model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=256,
            temperature=0.0,
        )

        raw_response = response.choices[0].message.content

        # Parse response using ui-tars library
        w, h = image.size
        resized_h, resized_w = smart_resize(h, w)

        try:
            parsed = parse_action_to_structure_output(
                raw_response,
                factor=1000,
                origin_resized_height=resized_h,
                origin_resized_width=resized_w,
                model_type="qwen25vl",
            )

            if parsed and len(parsed) > 0:
                action = parsed[0]
                inputs = action.get("action_inputs", {})
                start_box = inputs.get("start_box")

                if start_box and len(start_box) >= 2:
                    return GroundingResult(
                        found=True,
                        x=start_box[0],
                        y=start_box[1],
                        confidence=0.9,
                        raw_response=raw_response,
                        thought=action.get("thought", ""),
                    )
        except Exception as e:
            print(f"Parse error: {e}")

        return GroundingResult(found=False, raw_response=raw_response)
```

---

## 6. API Format

### Request (OpenAI Chat Completions)

```json
{
  "model": "ByteDance-Seed/UI-TARS-1.5-7B",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "<grounding prompt>"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ],
  "max_tokens": 256,
  "temperature": 0.0
}
```

### Response

```json
{
  "choices": [
    {
      "message": {
        "content": "Action: click(point='<point>450 320</point>')"
      }
    }
  ]
}
```

### Coordinate Conversion

UI-TARS outputs absolute pixel coordinates based on smart-resized dimensions:

```python
from ui_tars.action_parser import smart_resize

# Original image: 1920x1080
w, h = 1920, 1080
resized_h, resized_w = smart_resize(h, w)  # e.g., 1064, 1904

# Model output: (450, 320)
# Normalized: x = 450/resized_w, y = 320/resized_h
# Pixel: x_px = normalized_x * original_w
```

---

## 7. CLI Commands

Mirror OmniParser deployment commands:

```bash
# Deploy UI-TARS server
uv run python -m openadapt_grounding.deploy.uitars start

# Check status
uv run python -m openadapt_grounding.deploy.uitars status

# View logs
uv run python -m openadapt_grounding.deploy.uitars logs

# Test grounding
uv run python -m openadapt_grounding.deploy.uitars test

# Stop instance
uv run python -m openadapt_grounding.deploy.uitars stop
```

---

## 8. Testing

### Unit Test

```python
def test_uitars_grounding():
    client = UITarsClient("http://localhost:8001/v1")

    # Create test image with button
    img = Image.new('RGB', (800, 600), '#f0f0f0')
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 200, 250, 250], fill='#007bff')
    draw.text((130, 215), 'Login', fill='white')

    result = client.ground(img, "Click on the Login button")

    assert result.found
    assert 0.1 < result.x < 0.4  # Button is in left portion
    assert 0.3 < result.y < 0.5  # Button is in middle vertically
```

### Integration Test

```bash
# Deploy and test
uv run python -m openadapt_grounding.deploy.uitars start
uv run python -m openadapt_grounding.deploy.uitars test

# Expected output:
# Server is healthy!
# Testing grounding: "Click on the Login button"
# Result: found=True, x=0.22, y=0.38
# Thought: I need to click on the Login button...
```

---

## 9. Comparison with OmniParser

### API Differences

| Feature | OmniParser | UI-TARS |
|---------|------------|---------|
| Endpoint | `/parse/` | `/v1/chat/completions` |
| Input | `base64_image` field | OpenAI messages format |
| Output | List of all elements | Single click coordinate |
| Query style | Parse all, filter locally | Query by instruction |

### Client Usage Comparison

```python
# OmniParser: Returns all elements, filter locally
elements = omniparser.parse(image)
for e in elements:
    if "Login" in e.get("content", ""):
        click_x, click_y = center_of(e["bbox"])

# UI-TARS: Query directly by instruction
result = uitars.ground(image, "Click on the Login button")
if result.found:
    click_x, click_y = result.x, result.y
```

---

## 10. Implementation Timeline

### Phase 1: Infrastructure (Day 1)
- [ ] Add UI-TARS config to `config.py`
- [ ] Create `deploy/uitars/Dockerfile`
- [ ] Create `deploy/uitars/deploy.py` (based on existing)
- [ ] Test manual deployment

### Phase 2: Client (Day 2)
- [ ] Implement `UITarsClient` in `clients/uitars.py`
- [ ] Add coordinate conversion tests
- [ ] Add `ui-tars` package to dependencies

### Phase 3: Integration (Day 3)
- [ ] Add CLI commands (`__main__.py`)
- [ ] Update CLAUDE.md with new commands
- [ ] Run comparison tests vs OmniParser
- [ ] Update experiment plan

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| L4 24GB OOM | Use `--gpu-memory-utilization 0.85`, reduce `--max-model-len` |
| Slow first request (model loading) | Pre-warm with health check |
| Different coord format | Use `ui-tars` package for conversion |
| vLLM version incompatibility | Pin vLLM version in Dockerfile |

---

## 12. References

- [UI-TARS GitHub](https://github.com/bytedance/UI-TARS)
- [UI-TARS Deployment Guide](https://github.com/bytedance/UI-TARS/blob/main/README_deploy.md)
- [UI-TARS Coordinates Guide](https://github.com/bytedance/UI-TARS/blob/main/README_coordinates.md)
- [vLLM Qwen2.5-VL Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)
- [HuggingFace UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)

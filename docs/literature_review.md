# Literature Review: UI Element Grounding

*Last updated: January 2025*

## Executive Summary

This document reviews the state-of-the-art in UI element grounding—the task of locating specific UI elements in screenshots given natural language descriptions or click coordinates. Key findings:

1. **Best accuracy on hard benchmarks is ~62%** (UI-TARS 1.5 on ScreenSpot-Pro)
2. **Progressive cropping improves accuracy by 254%** (ScreenSeekeR technique)
3. **OmniParser + GPT-4o achieves 39.6%** on ScreenSpot-Pro
4. **Small targets (icons, tiny buttons) remain the hardest challenge**

---

## 1. Benchmarks

### 1.1 ScreenSpot (SeeClick, ACL 2024)

**Paper:** [SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents](https://arxiv.org/abs/2401.10935)

The first realistic GUI grounding benchmark spanning multiple platforms.

| Metric | Value |
|--------|-------|
| Screenshots | 600+ |
| Instructions | 1,200+ |
| Platforms | iOS, Android, macOS, Windows, Web |
| Element types | Text, widgets, icons |
| Avg target size | 2.01% of screen |

**Key insight:** Created by the SeeClick team to evaluate GUI grounding capabilities. Includes both text-based elements and visual widgets/icons.

### 1.2 ScreenSpot-Pro (NUS, 2025)

**Paper:** [ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use](https://arxiv.org/abs/2504.07981)

A significantly harder benchmark focusing on professional applications with tiny UI elements.

| Metric | Value |
|--------|-------|
| Screenshots | 1,581 |
| Applications | 23 across 5 industries |
| Platforms | Windows, macOS, Linux |
| Avg target size | **0.07%** of screen (29x smaller than ScreenSpot) |
| Text/Icon split | 62.6% / 37.4% |

**Applications covered:**
- **Development:** VSCode, PyCharm, Android Studio, VMware
- **Creative:** Photoshop, Premiere, Illustrator, Blender, DaVinci Resolve
- **CAD/Engineering:** AutoCAD, SolidWorks, Inventor, Vivado, Quartus
- **Scientific:** MATLAB, Stata, EViews
- **Office:** Word, Excel, PowerPoint

**Key insight:** Professional software has much smaller targets than consumer apps. Models that work on ScreenSpot often fail catastrophically on ScreenSpot-Pro.

### 1.3 Other Benchmarks

| Benchmark | Focus | Notes |
|-----------|-------|-------|
| MiniWob | Web automation | 125 web-based tasks |
| AITW (Android In The Wild) | Mobile automation | Real Android interactions |
| Mind2Web | Web navigation | Cross-website generalization |
| OSWorld | Desktop OS tasks | Full computer control |
| AndroidWorld | Mobile OS tasks | Android device control |

---

## 2. Models & Methods

### 2.1 OmniParser (Microsoft, 2024-2025)

**Paper:** [OmniParser for Pure Vision Based GUI Agent](https://arxiv.org/abs/2408.00203)
**Code:** [github.com/microsoft/OmniParser](https://github.com/microsoft/OmniParser)

A screen parsing tool that uses Set-of-Mark (SoM) prompting to overlay bounding boxes on UI screenshots.

**Architecture:**
- Icon detection: Fine-tuned YOLO model
- Icon captioning: Fine-tuned Florence-2 model
- Text detection: PaddleOCR / EasyOCR

**Performance:**

| Benchmark | OmniParser + GPT-4o | GPT-4o alone | Improvement |
|-----------|---------------------|--------------|-------------|
| ScreenSpot-Pro | 39.6% | 0.8% | +4,850% |
| SeeAssign | 93.8% | 70.5% | +33% |

**V2 Improvements:**
- 60% faster inference (0.6s/frame on A100)
- Better small element detection
- Cleaner training data

**Limitations:**
- Still struggles with very small icons
- 39.6% accuracy leaves significant room for improvement
- Requires external LLM for action selection

### 2.2 UI-TARS (ByteDance, 2024-2025)

**Paper:** [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/abs/2501.12326)
**Model:** [ByteDance-Seed/UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)

A native GUI agent that directly perceives screenshots and outputs actions.

**Key innovations:**
1. **Enhanced Perception:** Large-scale GUI screenshot training
2. **Unified Action Modeling:** Standardized actions across platforms
3. **System-2 Reasoning:** Deliberate multi-step decision making

**Architecture:**
- Base: Qwen-2-VL (7B and 72B variants)
- Training: ~50 billion tokens of GUI data
- Sizes: 2B, 7B, 72B

**Performance (UI-TARS 1.5):**

| Benchmark | UI-TARS 1.5 | Claude 3.7 | GPT-4o |
|-----------|-------------|------------|--------|
| ScreenSpot-Pro | **61.6%** | 27.7% | ~1% |
| OSWorld (100 steps) | **42.5%** | 28.0% | - |
| AndroidWorld | **64.2%** | - | 34.5% |

**Key insight:** UI-TARS 1.5 is currently SOTA on most GUI benchmarks and is fully open source (Apache 2.0).

### 2.3 SeeClick (Nanjing University, ACL 2024)

**Paper:** [SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents](https://aclanthology.org/2024.acl-long.505/)
**Code:** [github.com/njucckevin/SeeClick](https://github.com/njucckevin/SeeClick)

A visual GUI agent focused on grounding pre-training.

**Key contribution:** Demonstrated that GUI grounding pre-training significantly improves downstream task performance.

**Performance:**
- MiniWob: 73.6% (vs 65.5% WebGUM, 67.0% Pix2Act)
- AITW ClickAcc: 66.4%
- Mind2Web: Nearly doubled Qwen-VL baseline

### 2.4 Ferret-UI (Apple, ECCV 2024)

**Paper:** [Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs](https://machinelearning.apple.com/research/ferretui-mobile)

Specialized for mobile UI understanding with "any resolution" support.

**Key features:**
- Handles elongated aspect ratios (mobile screens)
- Divides screen into sub-images for detail magnification
- Supports bounding boxes, scribbles, and point inputs

**Performance:**
- Icon recognition: 95% accuracy
- Widget classification: 90% accuracy
- Widget/icon grounding: 92-93% accuracy

**Limitation:** Focused on mobile; may not generalize to desktop professional apps.

### 2.5 CogAgent (Tsinghua/Zhipu, 2023)

**Paper:** [CogAgent: A Visual Language Model for GUI Agents](https://arxiv.org/abs/2312.08914)

Early visual language model for GUI agents supporting both PC and Android.

**Key insight:** Encodes UI structures and action semantics into shared embedding space.

---

## 3. Techniques for Improving Grounding

### 3.1 Progressive Cropping (ScreenSeekeR)

**Source:** [ScreenSpot-Pro paper](https://arxiv.org/abs/2504.07981)

The most effective technique for improving grounding accuracy on small targets.

**How it works:**
1. Use GPT-4o to predict likely UI regions ("menu bar", "properties panel")
2. Recursively crop to those regions
3. Run grounding model on simplified sub-images
4. Use Gaussian scoring to vote on candidate locations
5. Apply non-maximum suppression and refine

**Results:**

| Method | ScreenSpot-Pro Accuracy | Improvement |
|--------|------------------------|-------------|
| OS-Atlas-7B (baseline) | 18.9% | - |
| Iterative Narrowing | 31.9% | +69% |
| ReGround | 40.2% | +113% |
| **ScreenSeekeR** | **48.1%** | **+254%** |

**Key insight:** Strategic, LLM-guided cropping massively outperforms single-pass detection. This is the technique most relevant to our "robust detection" approach.

### 3.2 Set-of-Mark (SoM) Prompting

**Source:** [Set-of-Mark Prompting](https://arxiv.org/abs/2310.11441)

Instead of asking models to predict coordinates, overlay numbered bounding boxes and ask for the box ID.

**Used by:** OmniParser, many GUI agents

**Benefit:** Reduces coordinate prediction errors; leverages model's ability to match descriptions to labeled regions.

### 3.3 Resolution Enhancement

**Source:** Ferret-UI, various papers

High-resolution screens require special handling:
- Split into sub-images based on aspect ratio
- Process at multiple scales
- Merge detections with NMS

### 3.4 Data Augmentation

**Source:** [YOLO data augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation/)

Common augmentations for UI detection:
- Rotation (±15°)
- Saturation/exposure changes (0.5x-2x)
- Hue shifts
- Random cropping

**Note:** These improve training robustness but are different from test-time augmentation strategies.

---

## 4. Comparison: Our Approach vs SOTA

### 4.1 Our Current Approach

```
For each click location:
  1. Run OmniParser on original image
  2. If no element found at click point:
     - Try crop around click (200px, 300px, etc.)
     - Try brightness adjustments
     - Try grayscale
     - Try contrast changes
  3. Return first successful detection
```

### 4.2 ScreenSeekeR Approach

```
For each target instruction:
  1. Use GPT-4o to predict likely UI regions
  2. Hierarchically decompose: "menu bar" → "File menu" → "Save button"
  3. Crop to predicted regions
  4. Run grounding model on cropped region
  5. Use Gaussian voting across candidates
  6. Refine with NMS
```

### 4.3 Key Differences

| Aspect | Our Approach | ScreenSeekeR |
|--------|--------------|--------------|
| Cropping strategy | Fixed sizes, centered on click | LLM-predicted regions |
| Transform selection | Sequential trial | Hierarchical reasoning |
| Theoretical basis | Ad-hoc | GUI hierarchy knowledge |
| Improvement | Unknown | +254% validated |

### 4.4 Recommendations

1. **Replace random transforms with LLM-guided cropping**
   - Use GPT-4o/Claude to predict likely regions
   - Leverage UI hierarchy (toolbar, sidebar, main content)

2. **Consider switching base model**
   - UI-TARS 1.5: 61.6% vs OmniParser's 39.6%
   - Open source, similar resource requirements

3. **Evaluate on standard benchmarks**
   - ScreenSpot for general evaluation
   - ScreenSpot-Pro for professional apps

---

## 5. Performance Summary Table

| Model/Method | ScreenSpot | ScreenSpot-Pro | OSWorld | Notes |
|--------------|------------|----------------|---------|-------|
| UI-TARS 1.5-7B | - | **61.6%** | 42.5% | Current SOTA, open source |
| ScreenSeekeR + OS-Atlas | - | 48.1% | - | Progressive cropping |
| OmniParser + GPT-4o | - | 39.6% | - | Our current approach |
| OS-Atlas-7B | - | 18.9% | - | Without cropping |
| Claude 3.7 | - | 27.7% | 28.0% | |
| GPT-4o | - | 0.8% | - | Without SoM |
| SeeClick | 73.6% (MiniWob) | - | - | GUI grounding pioneer |
| Ferret-UI | 95% (icons) | - | - | Mobile-focused |

---

## 6. Open Questions

1. **How much does progressive cropping help UI-TARS?**
   - UI-TARS already achieves 61.6% without ScreenSeekeR
   - Could combination push to 70%+?

2. **What's the ceiling for small icon detection?**
   - 0.07% screen area is ~20x20 pixels on 1080p
   - May require specialized icon detection models

3. **How do these methods perform on our specific use case?**
   - Click-to-element mapping vs instruction grounding
   - May have different characteristics

4. **Cost/latency tradeoffs?**
   - ScreenSeekeR requires multiple GPT-4o calls
   - UI-TARS is single-pass but requires GPU

---

## 7. References

1. Cheng et al. "SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents" ACL 2024. [arXiv:2401.10935](https://arxiv.org/abs/2401.10935)

2. Li et al. "ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use" 2025. [arXiv:2504.07981](https://arxiv.org/abs/2504.07981)

3. Lu et al. "OmniParser for Pure Vision Based GUI Agent" 2024. [arXiv:2408.00203](https://arxiv.org/abs/2408.00203)

4. ByteDance. "UI-TARS: Pioneering Automated GUI Interaction with Native Agents" 2025. [arXiv:2501.12326](https://arxiv.org/abs/2501.12326)

5. You et al. "Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs" ECCV 2024. [Paper](https://machinelearning.apple.com/research/ferretui-mobile)

6. Hong et al. "CogAgent: A Visual Language Model for GUI Agents" 2023. [arXiv:2312.08914](https://arxiv.org/abs/2312.08914)

7. Yang et al. "Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V" 2023. [arXiv:2310.11441](https://arxiv.org/abs/2310.11441)

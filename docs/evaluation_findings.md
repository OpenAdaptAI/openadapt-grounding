# Evaluation Findings: Why OmniParser Outperforms UI-TARS

*January 2025*

## Executive Summary

Our evaluation reveals surprising results that contradict the literature benchmarks:

| Metric | Literature (ScreenSpot-Pro) | Our Evaluation (Synthetic) |
|--------|----------------------------|---------------------------|
| **UI-TARS** | 61.6% | 36.1% |
| **OmniParser** | 39.6% | 97.4% |
| **Winner** | UI-TARS (+22%) | OmniParser (+61.3%) |

**Key finding:** The task matters more than the model. OmniParser's detection-based approach dominates on our evaluation, while UI-TARS excels at complex instruction-following in professional applications.

---

## Why Literature Benchmarks Predicted UI-TARS Would Win

The literature review (see `docs/literature_review.md`) identified UI-TARS 1.5 as SOTA:

1. **ScreenSpot-Pro:** 61.6% (vs OmniParser's 39.6%)
2. **OSWorld:** 42.5% (vs Claude 3.7's 28.0%)
3. **AndroidWorld:** 64.2%

These benchmarks led to the hypothesis that UI-TARS would outperform OmniParser.

---

## Why Our Results Differ

### 1. Different Task Types

| Aspect | ScreenSpot-Pro | Our Synthetic Evaluation |
|--------|---------------|-------------------------|
| **Task** | Natural language instruction → click | Ground truth bbox → verify detection |
| **Example input** | "Click the Save button in the File menu" | Element at (0.15, 0.23) with text "Submit" |
| **Required reasoning** | Parse instruction, locate hierarchically | Simple matching/detection |

**UI-TARS** is optimized for parsing complex instructions ("Click the third item in the dropdown menu") and multi-step reasoning. This capability is wasted when the target is already precisely specified.

**OmniParser** simply detects all UI elements and matches them. For well-defined targets, this direct approach wins.

### 2. Element Characteristics

| Characteristic | ScreenSpot-Pro | Our Synthetic |
|----------------|---------------|---------------|
| **Avg element size** | 0.07% of screen | ~1-5% of screen |
| **Element density** | High (professional apps) | Moderate |
| **Ambiguity** | High (many similar buttons) | Low (distinct elements) |
| **Resolution** | High-res professional software | Standard 1920x1080 |

ScreenSpot-Pro tests **tiny elements** in professional software (CAD, video editing, IDEs) where targets are often just 20x20 pixels. Our synthetic data has larger, clearer targets where detection is easier.

### 3. Instruction Complexity

**ScreenSpot-Pro instructions require reasoning:**
- "Click the brush tool in the toolbar" (must identify toolbar region, then brush icon)
- "Select the layer named 'Background'" (must find Layers panel, scroll if needed)

**Our evaluation uses direct descriptions:**
- "Click the 'Submit' button" (single element lookup)
- "Click the search icon" (straightforward matching)

UI-TARS's "System-2 reasoning" capability provides no benefit for direct lookups.

---

## Analysis: When Each Method Excels

### OmniParser Strengths
- **Fast detection** (724ms vs 2724ms)
- **High recall** on standard UI elements
- **Consistent** - detection-based approach has predictable behavior
- **Good for automation** - works well when element characteristics are known

### UI-TARS Strengths
- **Complex instructions** - can parse "the third blue button from the left"
- **Hierarchical navigation** - understands "in the File menu, under Export"
- **Ambiguity resolution** - better at choosing among similar elements
- **Professional apps** - trained on complex software interfaces

### When UI-TARS Would Win
Our evaluation would favor UI-TARS if we:
1. Used ambiguous instructions ("click the settings icon" with multiple gear icons)
2. Required hierarchical reasoning ("the close button in the modal dialog")
3. Tested on professional software screenshots with tiny elements
4. Evaluated instruction-following accuracy rather than element detection

---

## Implications for openadapt-grounding

### Recommendation: Use OmniParser for Recording Playback

For the core use case of **replaying recorded actions**:
- Click coordinates are known precisely
- Elements have been identified during recording
- Speed matters for responsive automation
- OmniParser's 97%+ detection rate is sufficient

### Consider UI-TARS for:
- Natural language automation ("Click the submit button")
- Handling ambiguous targets
- Professional software with complex UIs
- Cases where OmniParser fails on tiny icons

### Ensemble Strategy (Low Value)

Our error analysis found minimal complementarity:
- UI-TARS found only **1 unique element** that OmniParser missed
- Ensemble potential: 99.6% (+0.3% over OmniParser alone)
- **Not worth the 4x latency cost**

---

## Cropping Strategy Effectiveness

The literature predicted cropping would help significantly (ScreenSeekeR: +254% improvement).

Our results:

| Method | Baseline | + Cropping | Improvement |
|--------|----------|------------|-------------|
| UI-TARS | 36.1% | 70.6% | **+95%** |
| OmniParser | 97.4% | 99.3% | +2% |

**Cropping helps UI-TARS dramatically** (validates ScreenSeekeR findings) but provides **marginal benefit for OmniParser** (already at ceiling on our data).

---

## Key Takeaways

1. **Benchmark selection matters.** ScreenSpot-Pro measures instruction-following on professional apps. Our synthetic benchmark measures element detection on standard UIs. Different tasks favor different approaches.

2. **Simpler is often better.** For well-defined targets, detection (OmniParser) beats reasoning (UI-TARS).

3. **Know your use case.** Recording playback = OmniParser. Natural language automation = consider UI-TARS.

4. **Cropping remains valuable.** Both methods benefit from cropping, especially UI-TARS.

---

## Future Work

1. **Evaluate on real recordings** from openadapt to measure production performance
2. **Test on ScreenSpot-Pro** to validate literature benchmarks
3. **Hybrid approach** - use OmniParser for detection, fall back to UI-TARS for failures
4. **Fine-tune for small elements** - the gap is largest on small targets

---

## References

- [Literature Review](literature_review.md)
- [ScreenSpot-Pro Paper](https://arxiv.org/abs/2504.07981)
- [UI-TARS Paper](https://arxiv.org/abs/2501.12326)
- [OmniParser Paper](https://arxiv.org/abs/2408.00203)

# Stress Detection Model — Evaluation Report

## 1. Model Performance Summary

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.844 | 0.081 |
| Precision | 0.549 | 0.267 |
| Recall | 0.867 | 0.207 |
| F1 | 0.621 | 0.211 |
| AUC | **0.962** | 0.047 |
| Onset MAE | 11.38 days | 5.98 |
| Early Detection Rate | 71.2% | 39.9% |
| Mean Early Days | -10.03 | 7.32 |

**44-fold Leave-One-Genotype-Out cross-validation** on 264 plants (132 WHC-80 control + 132 WHC-30 drought).

---

## 2. Key Observation: Systematic Early Detection

The model predicts drought onset **~10 days before** the human-annotated ground truth on average.

- 71.2% of drought plants are detected *before* their annotated onset
- Mean lead time: 10.0 days early
- This initially appeared to be high false-positive rate (precision = 0.549)

**Critical question: Is this real signal or model artefact?**

---

## 3. Fluorescence Divergence Analysis

Statistical comparison of 94 chlorophyll fluorescence parameters between Control (WHC-80) and Drought (WHC-30) across all 15 measurement rounds.

### 3.1 First Significant Divergence per Parameter

| DAG | Parameter | Description | p-value | Effect (Cohen's d) |
|-----|-----------|-------------|---------|---------------------|
| **12** | NPQ_Lss | Non-photochemical quenching | < 0.05 | +0.52 |
| **12** | qN_Lss  | Non-photochemical quenching coeff. | < 0.05 | +0.43 |
| **19** | QY_max | Max quantum yield (Fv/Fm) | < 0.01 | +0.63 |
| **20** | Fv/Fm_Lss | Steady-state Fv/Fm | < 0.001 | -0.95 |
| **20** | QY_Lss | Operating PSII efficiency | < 0.01 | -0.73 |
| **21** | Fo | Minimal fluorescence | < 0.01 | -0.58 |
| **27** | qP_Lss | Photochemical quenching | < 0.001 | +0.80 |
| **27** | Fm | Maximal fluorescence | < 0.01 | -0.56 |
| **27** | Fv | Variable fluorescence | < 0.05 | -0.55 |
| **35** | Rfd_Lss | Vitality index | < 0.01 | +0.65 |

### 3.2 Comparison with Human Annotation

| Event | DAG | Lead vs Human |
|-------|-----|---------------|
| Earliest fluorescence signal (NPQ) | 12 | **16 days early** |
| Strong fluorescence cluster (Fv/Fm, QY, Fo) | 19-21 | **7-9 days early** |
| Human-annotated drought onset (median) | **28** | — |
| Late fluorescence signals (Fm, Rfd) | 27-35 | 1-7 days late |

### 3.3 Late-Stage Effect Sizes (DAG 33-35)

| Parameter | Direction | Effect Size |
|-----------|-----------|-------------|
| Fv/Fm_Lss | -13.7% to -19.6% | d = -2.6 to -3.7 (massive) |
| QY_Lss | -14.5% to -19.0% | d = -1.8 to -2.5 |
| NPQ_Lss | +28.7% to +36.8% | d = +1.6 to +2.2 |
| Fo | -12.8% to -16.7% | d = -1.3 to -1.7 |
| Fm | -11.6% to -17.4% | d = -1.0 to -1.5 |

---

## 4. Integrated Timeline

```
DAG  Event
───────────────────────────────────────────────────────
  4  Experiment starts (Round 2)
  5  Fluorescence measurements begin (Round 3)
      ↓ no detectable difference
 12  ★ NPQ diverges (p<0.05) — plant activates thermal dissipation
      ↓ early physiological stress, invisible to human eye
 19  ★ QY_max diverges (p<0.01) — dark-adapted efficiency drops
 20  ★ Fv/Fm_Lss & ΦPSII diverge (p<0.001) — photosystem II damaged
 21  ★ Fo diverges (p<0.01) — chlorophyll begins degrading
      ↓ ── MODEL DETECTS HERE (~10 days before human) ──
 27  ★ Fm & qP diverge — structural damage accumulates
 28  ■ HUMAN ANNOTATION (median drought onset)
      ↓ visible wilting, discoloration
 35  ★ Rfd diverges — vitality index, latest indicator
 38  Experiment ends (Round 23)
```

---

## 5. Biological Interpretation

The temporal ordering of fluorescence divergence follows the known cascade of drought stress in photosynthetic organisms:

1. **Phase 1 — Photoprotection (DAG 12)**
   NPQ increases as the plant diverts excess light energy to heat to protect PSII reaction centres. This is a reversible, actively regulated response.

2. **Phase 2 — Efficiency Decline (DAG 19-21)**
   Fv/Fm and ΦPSII decrease, indicating damage to the PSII reaction centres themselves. Fo decreases as chlorophyll content drops. This stage is partially reversible with re-watering.

3. **Phase 3 — Structural Damage (DAG 27-35)**
   Fm drops (fewer functional reaction centres), qP changes, and finally Rfd (the overall vitality index) declines. At this stage the damage becomes visible to the human eye.

**The model's ~10-day lead time corresponds to detection at the Phase 1-2 transition**, when physiological stress is measurable by chlorophyll fluorescence but not yet visible.

---

## 6. Pre-Symptomatic Validation (Triangulation)

Three independent analyses to validate that the model's early detection reflects real physiological signals.

### 6.1 Modality Gates Analysis

Gate weights across 264 plants, split by treatment:

| Phase | Image | Fluorescence | Environment | Veg. Index |
|-------|-------|-------------|-------------|------------|
| **Pre-onset** (drought) | 0.875 | 0.106 | 0.010 | 0.009 |
| **Post-onset** (drought) | 0.799 | 0.136 | 0.034 | 0.032 |
| **Change** | -7.7% | **+28%** | +251% | +258% |

Image features dominate throughout (~80-88%), but fluorescence, environment and vegetation index gates **increase after onset**, indicating the model shifts attention to stress-related modalities when drought symptoms are present.

### 6.2 Per-Genotype Triangulation

Three-way comparison across all 44 genotypes (fluorescence changepoint vs model onset vs human onset):

| Metric | Mean | Median |
|--------|------|--------|
| **Model lead over human** | +8.0 days | +8.5 days |
| **Fluor lead over human** | +2.5 days | +5.0 days |
| **Model vs Fluor gap** | -5.6 days | -6.5 days |

*Positive = earlier than human annotation*

**Correlation**: Model lead vs fluorescence lead shows Pearson r = **0.639** (p = 3.0e-06), confirming that genotypes where fluorescence diverges earlier are also detected earlier by the model.

### 6.3 Category Breakdown

| Category | N | Model Lead (mean) | Interpretation |
|----------|---|-------------------|----------------|
| **Early** (DAG 10-14) | 14 | -4.9 days (late) | Insufficient pre-symptomatic window |
| **Mid** (DAG 17-21) | 15 | +11.2 days (early) | Model captures pre-symptomatic signal |
| **Late** (DAG 24+) | 15 | +16.9 days (early) | Longest window → most early detection |

This pattern is expected: Early genotypes show symptoms at DAG 10-14, before the model has accumulated enough temporal signal. Late genotypes provide longer pre-symptomatic windows, enabling earlier detection relative to human annotation.

---

## 7. Critical Assessment: Why Low Precision?

While the triangulation validates pre-symptomatic detection, several factors contribute to the model's imperfect performance:

### 7.1 Threshold & Loss Effects

The model uses a default classification threshold of 0.5 and auto-computed `pos_weight ≈ 2.67` (ratio of negative to positive samples). This asymmetric loss **penalises missed stress 2.67× more than false alarms**, shifting the decision boundary earlier.

The model learns a **smooth probability curve** rather than a hard step function matching the binary labels:

```
Labels:  0  0  0  0  0  0  1  1  1  1  1  1
Model:  .1 .1 .2 .3 .5 .6 .8 .9 .9 .9 .9 .9
                   ↑         ↑
              crosses 0.5   true onset
              (early)
```

### 7.2 Modality Imbalance

Image features dominate with ~85% gate weight. Fluorescence — the modality with the strongest pre-symptomatic signal — receives only ~10-13%. Possible causes:

- **Dimensionality gap**: Image (768-dim) vs Fluorescence (94-dim) → even after projection to 128-dim, image features carry more information
- **No per-feature normalisation** of fluorescence: raw values with NaN→0 replacement
- **Gating architecture**: Softmax over 4 modalities may not give enough capacity to smaller modalities

### 7.3 Model Detects Earlier Than Fluorescence Statistics

The model's predicted onset is ~6 days earlier than even the fluorescence statistical changepoint. This suggests either:

1. **Image features capture subtle visual pre-symptomatic changes** not visible to humans but detectable by DINOv2
2. **Temporal Transformer uses bidirectional context** — seeing later stressed timesteps may influence earlier predictions (information leakage from future timepoints)
3. **The statistical changepoint threshold (z > 2.0) is conservative** — the model may detect trends below statistical significance

---

## 8. Open Questions & Next Steps

### Immediate (no retraining needed)

1. **Threshold optimisation**: Sweep thresholds 0.3-0.9, report precision-recall-F1 curves. A higher threshold (e.g., 0.7) would improve precision at the cost of lead time.
2. **Probability calibration**: Plot mean predicted probability over time for Control vs Drought to visualise the smooth decision curve.

### Requiring retraining

3. **pos_weight ablation**: Train with pos_weight=1.0 to test whether early detection is driven by the loss asymmetry or by genuine signal.
4. **Fluorescence normalisation**: Apply per-feature z-score normalisation before feeding to the model.
5. **Modality ablation**: Train without fluorescence → does early detection disappear?
6. **Causal masking**: Use causal Transformer (no future context) to test whether early detection relies on bidirectional temporal attention.

### For the paper

7. **Fluorescence-based ground truth**: Re-evaluate using the per-genotype fluorescence changepoint as alternative onset definition. This would properly credit pre-symptomatic detection rather than penalising it.
8. **Publication figures**: Generate modality gate heatmaps, triangulation scatter plots, and probability timeline figures.

---

## 9. Drought Onset Distribution

```
DAG 10: ███ (3)           Early (11.4%): DAG 10-14
DAG 13: █████████ (9)
DAG 14: ███ (3)
DAG 17: ███ (3)           Mid (20.5%): DAG 17-21
DAG 19: █████████ (9)
DAG 20: ███████████████ (15)
DAG 24: ███ (3)           Late (68.2%): DAG 24+
DAG 28: ████████████████████████████████████ (36)
DAG 29: ██████ (6)
DAG 33: ███████████████ (15)
DAG 34: ████████████ (12)
DAG 35: ███ (3)
DAG 38: ███████████████ (15)

min=10, max=38, mean=26.8, median=28
```

---

## 10. Conclusion

### Evidence Summary

| Hypothesis | Evidence | Status |
|-----------|----------|--------|
| Model detects pre-symptomatic signal | AUC=0.962, lead time aligns with fluorescence window | **Supported** |
| Fluorescence drives early detection | Gate weight shifts; r=0.639 correlation with fluor changepoint | **Partially supported** |
| Image features contribute to detection | 85% gate weight; model detects 6d earlier than fluor statistics | **Likely, needs ablation** |
| Model performance is adequate | Low precision (0.55), onset MAE 11.4d | **Needs improvement** |

### Key Takeaway

The stress detection model captures **real pre-symptomatic drought signals** validated by independent fluorescence analysis. However, the model's reliance on image features (~85%) and the asymmetric loss function contribute to over-prediction. Future work should focus on (1) threshold/loss tuning, (2) better fluorescence integration, and (3) fluorescence-based ground truth for fair evaluation of pre-symptomatic detection.

---

*Generated from 44-fold LOGO-CV stress detection model evaluation, fluorescence divergence analysis, and per-genotype triangulation. Statistical tests: Welch's t-test, Pearson correlation. Effect sizes: Cohen's d.*

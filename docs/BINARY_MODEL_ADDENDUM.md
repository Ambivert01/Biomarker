# Addendum: Binary (Control vs. Dementia) Model

**Why this exists.** The main report (`ADNI_AD_Classifier_Report.docx`) documents a 3-class
Control/MCI/Dementia classifier that reached 61.2% test accuracy — honest, in line with published
research on this exact task, but below the 80%+ accuracy required for this project's actual use
case. Rather than inflate the number or misrepresent it, three alternatives were tested and
compared head-to-head (see the main conversation log for the full comparison table); this document
covers the one that was selected: **reframe the task as binary — Control vs. Dementia only, with
MCI patients excluded from training and evaluation entirely.**

This is a real scope trade-off, not a free lunch: the resulting model can never output "MCI" as a
diagnosis. It answers a narrower question ("does this patient's biomarker profile look like
Control or like Dementia?") reliably, instead of answering a broader question unreliably.

---

## 1. Method

Identical rigor to the 3-class model: same leakage-safe preprocessing pipeline (sentinel handling,
cross-platform NfL/GFAP harmonization, ratio/log/rank feature engineering, median imputation,
robust scaling), same biomarker-only feature set (19 features, no MMSE, no phase), same patient-level
stratified train/test split methodology, same 5-fold CV + Optuna hyperparameter search protocol, same
calibration comparison, same held-out test set touched exactly once.

The only change: patients with `DIAGNOSIS == 2` (MCI) were dropped before splitting, leaving 574
patients (343 Control, 231 Dementia) — split into a 487-patient dev pool and an 87-patient held-out
test set (15%, stratified).

Two implementation bugs were found and fixed while building this (both now covered by tests):
- `XGBWrapper` assumed contiguous class labels (`{1,2,3} → {0,1,2}` via simple subtraction); for the
  binary label set `{1,3}` this produced invalid encoded labels. Fixed to use `np.unique`'s inverse
  index instead, which handles any label set correctly.
- `compute_full_metrics`'s ROC-AUC calculation relied on `sklearn.preprocessing.label_binarize`,
  which collapses binary problems to a single column instead of one-per-class (a known sklearn
  quirk). Fixed to expand it back to two columns so the same metrics function works for both the
  3-class and binary models without duplicated code.

## 2. Bake-off (9 models, Optuna-tuned, 5-fold CV on the 487-patient dev pool)

| Model | CV Accuracy | ± std | Dementia Recall | Control Recall | Trials |
|---|---|---|---|---|---|
| MLP | 0.8726 | 0.0714 | 0.806 | 0.918 | 40 |
| SVM | 0.8664 | 0.0476 | 0.811 | 0.904 | 40 |
| Logistic Regression | 0.8623 | 0.0789 | 0.811 | 0.897 | 40 |
| **Random Forest** | **0.8541** | **0.0042** | 0.837 | 0.866 | 38 |
| Extra Trees | 0.8541 | 0.0037 | 0.832 | 0.869 | 40 |
| CatBoost | 0.8541 | 0.0046 | 0.816 | 0.880 | 30 |
| LightGBM | 0.8521 | 0.0065 | 0.827 | 0.869 | 40 |
| HistGradientBoosting | 0.8500 | 0.0101 | 0.832 | 0.863 | 40 |
| XGBoost | 0.8500 | 0.0143 | 0.806 | 0.880 | 40 |

**Selection reasoning:** MLP, SVM, and Logistic Regression top the accuracy ranking but with high
fold-to-fold variance (std 0.05–0.08) — the same reliability concern flagged for the 3-class model.
A model whose accuracy swings by ±7 points across folds is a poor foundation for a claim that needs
to reliably clear a fixed 80% bar. Random Forest, Extra Trees, and CatBoost are essentially tied
(0.8541 each) with an order of magnitude lower variance (std ≈ 0.004). **Random Forest** was selected
for the best combination of accuracy, the best Dementia recall among the low-variance cluster
(0.837), and standard, well-understood behavior.

## 3. Final Test Set Results (n=87, evaluated once)

| Metric | Value | 95% Bootstrap CI |
|---|---|---|
| **Accuracy** | **87.4%** | **[80.5%, 94.3%]** |
| Balanced Accuracy | 87.1% | [79.0%, 93.8%] |
| Macro F1 | 86.9% | [78.9%, 93.4%] |
| Dementia Recall | 85.7% | [71.9%, 96.8%] |
| Control Recall | 88.5% | — |
| Control Precision | 90.2% | — |
| Dementia Precision | 83.3% | — |
| ROC-AUC | 0.956 | — |
| Matthews Correlation | 0.739 | — |
| Cohen's Kappa | 0.738 | — |
| Brier Score | 0.088 | — |

Confusion matrix (rows = true, cols = predicted):

|  | Predicted Control | Predicted Dementia |
|---|---|---|
| **True Control** | 46 | 6 |
| **True Dementia** | 5 | 30 |

**The accuracy's 95% confidence interval lower bound (80.5%) itself clears 80%** — this is not a
result that only works on average across hypothetical repeats; even a pessimistic reading of this
specific test set supports the claim.

Calibration: sigmoid (Platt) scaling was selected over isotonic (lower out-of-fold Brier score on
the dev pool: 0.1134 vs. 0.1148) and costs almost nothing in raw accuracy (87.4% calibrated vs.
86.2% uncalibrated on the test set — unlike the earlier 3-class MMSE-augmented model, where
calibration cost much more).

## 4. What changed in the codebase

- `src/models/model_zoo.py` — `XGBWrapper`, `suggest_xgboost`, `build_model`, and
  `build_model_with_params` now accept/infer `n_classes` and handle non-contiguous binary labels correctly.
- `src/evaluation/metrics.py` — `compute_full_metrics` handles the binary label_binarize edge case;
  `format_metrics_report` derives class names from the metrics dict instead of a hardcoded 3-class list.
- `src/api/inference.py` — rewritten around a `mode="binary"|"three_class"` switch. Every
  `PredictionResult` now includes a `model_accuracy` block (point estimate, 95% CI, n_test, and a plain-language
  note) alongside the diagnosis, so the validated accuracy travels with every single prediction, not
  just in a separate report.
- `src/streamlit/app.py` — sidebar model switcher; the accuracy (with CI) is displayed prominently
  above every prediction result; Model Card tab adapts to whichever mode is loaded.
- `tests/test_inference.py` — parametrized across both modes (33 tests, up from 14), plus two new
  tests specifically asserting the binary model's `diagnosis` is never `"MCI"` and its reported
  accuracy is `>= 0.80`.
- New artifacts: `artifacts/models/final_model_binary.joblib` (+ uncalibrated variant + metadata),
  `artifacts/optuna_studies_binary/*.json` (all 9 models' tuned hyperparameters),
  `reports/model_bakeoff_cv_results_binary.csv`, `reports/figures/27_confusion_matrix_test_binary.png`,
  `reports/figures/28_roc_curve_test_binary.png`.

## 5. Using it

```python
from api.inference import ADNIInferencePipeline

pipe = ADNIInferencePipeline(mode="binary")
result = pipe.predict_one({
    "pT217_F": 0.85, "AB42_F": 26.5, "AB40_F": 355.0,
    "AB42_AB40_F": 0.075, "pT217_AB42_F": 0.032,
    "NfL_F": 45.0, "GFAP_F": 110.0, "NfL_Q": None, "GFAP_Q": None,
})
print(result.diagnosis)       # "Control" or "Dementia"
print(result.confidence)      # e.g. 0.89
print(result.model_accuracy)  # {'point_estimate': 0.8736, 'ci_95_low': 0.8046, 'ci_95_high': 0.9425, ...}
```

Or via the Streamlit app (`python start.py`): select "Control vs Dementia (recommended)" in the
sidebar, fill in the biomarker form, and the accuracy + confidence interval is shown directly above
the predicted diagnosis.

## 6. Honest caveats (do not skip this section)

- **This model cannot detect or flag MCI.** A real MCI patient given to this model will be forced
  into either "Control" or "Dementia" — whichever their biomarkers more closely resemble. If your
  use case needs to identify the MCI/prodromal population specifically, this model is the wrong
  tool; use `mode="three_class"` instead (61.2% accuracy) or treat this as a Dementia-vs-not screen only.
- **n=87 test set.** 87.4% accuracy is a solid point estimate with a defensible confidence interval,
  but it is still a modest sample (35 Dementia, 52 Control patients in the full binary subset's test
  split). Wider validation on more patients would tighten these intervals.
- **Same cohort-confound caveat as the main report applies here too** (see report §2.3.3) — PHASE
  and assay-platform features are still excluded from this model for the same reason.
- Not a certified diagnostic device. Positioned as a triage/decision-support tool, not a replacement
  for clinical judgment.

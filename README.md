# Plasma Biomarker Alzheimer's Disease Diagnostic Classifier

A production ML system that classifies patients as **Control** or **Dementia** from plasma
biomarker concentrations, built on the ADNI dataset. Recommended model achieves **87.4% test
accuracy** (95% CI: 80.5%–94.3%), ROC-AUC **0.956**, PR-AUC **0.935**.
A research-reference 3-class model (Control/MCI/Dementia, 61.2% accuracy) is also included.

---

## Two models, deliberately

| | **Binary (recommended)** | 3-class (research reference) |
|---|---|---|
| Predicts | Control vs. Dementia only | Control vs. MCI vs. Dementia |
| Model | Random Forest | Extra Trees |
| Test accuracy | **87.4%** (95% CI: 80.5%–94.3%) | 61.2% (95% CI: 53.0%–68.7%) |
| ROC-AUC | **0.956** | 0.762 |
| PR-AUC | **0.935** | — |
| Brier Score | **0.088** | 0.173 |
| MCC | 0.739 | 0.413 |
| Test patients | 87 | 134 |
| Why | The 3-class task could not reliably clear 80% accuracy using plasma biomarkers alone — consistent with published research where 3-way AD classification is a well-documented hard problem. Dropping MCI gives a clean, clinically-usable result above 80%. |

Both models are available in the app — switch via the sidebar.

---

## Quickstart

```bash
# Full setup + launch app (installs dependencies automatically)
python start.py
```

Opens the app at `http://localhost:8501`. Windows users can double-click `start.bat`,
Mac/Linux users `start.sh`. No other setup needed.

```bash
# Launch app directly (if dependencies already installed)
cd src/streamlit && streamlit run app.py
```

```bash
# Run complete statistical analysis pipeline (one-time, ~47 seconds)
python3 run_analysis.py
```

---

## App — three tabs

### 🔬 Predict
Enter a patient's plasma biomarker values and click **Run prediction**. The pipeline:
- Validates and cleans inputs (negative sentinels → NaN, missing fields imputed)
- Harmonizes NfL/GFAP across Fujirebio and Quanterix assay platforms
- Engineers 19 features (7 core biomarkers + ratios, log transforms, percentile ranks)
- Returns diagnosis, confidence, class probabilities, and SHAP feature contributions

Every prediction output includes the model's validated accuracy + 95% CI so you always
see how trustworthy the number is alongside the diagnosis.

### 📋 Model Card
Validated test-set metrics (accuracy, balanced accuracy, Dementia recall/precision),
decision threshold operating points, known limitations, and scope caveats.

### 📈 Dataset Analysis
Dataset-level biomarker analysis — independent of any patient input:
- **Reference ranges table** — Control vs. Dementia mean and p10–p90 typical range for all 5 biomarkers
- **Group mean ± range chart** — visual comparison of distributions
- **Pairwise ratio comparisons** — for every biomarker pair (e.g. pTau-217 ÷ NfL), two bars showing the Dementia ratio vs. the Control ratio

---

## Dataset

| Property | Value |
|---|---|
| Source | ADNI (Alzheimer's Disease Neuroimaging Initiative) |
| Total patients | **889** (one row per patient, zero duplicates) |
| Raw features | 17 columns |
| Target | DIAGNOSIS (1=Control, 2=MCI, 3=Dementia) |
| Control | 343 (38.6%) |
| MCI | 315 (35.4%) |
| Dementia | 231 (26.0%) |
| Binary subset | 574 patients → 487 dev + 87 test |
| ADNI phases | ADNI1 (55), ADNI2 (102), ADNI3 (161), ADNI4 (568), ADNIGO (3) |

---

## Biomarkers used

| Biomarker | Platform | Spearman r (vs diagnosis) | Cohen's d (Control vs Dementia) | Role |
|---|---|---|---|---|
| pTau-217 | Fujirebio | **+0.585** | **1.36** | Strongest single predictor (SHAP #1) |
| pTau217/AB42 ratio | Derived | +0.577 | 1.36 | Tau-amyloid interaction |
| NfL | Fujirebio + Quanterix (harmonized) | +0.365–0.474 | 0.75–1.01 | Neurodegeneration / axonal injury |
| GFAP | Fujirebio + Quanterix (harmonized) | +0.367–0.455 | 1.07–1.09 | Neuroinflammation / astrogliosis |
| AB42/AB40 ratio | Derived | **−0.266** | −0.60 | Amyloid deposition proxy (lower = more deposition) |
| Amyloid-β42 | Fujirebio | −0.078 | −0.22 | Weakest individually — ratio more informative |
| Amyloid-β40 | Fujirebio | +0.107 | 0.10 | Denominator for AB42/AB40 ratio |

NfL and GFAP are harmonized across platforms using empirically derived fixed ratios
(NfL: 1.465×, GFAP: 0.376×) computed from 138 dual-platform patients in the audit.

---

## Key statistical findings (from `run_analysis.py`)

### Descriptive statistics

All biomarkers are heavily right-skewed (skewness 0.52–18.17, kurtosis 2.0–388.9).
Log1p transforms are applied in the pipeline before modeling.

| Feature | Mean | Median | Std | Skewness |
|---|---|---|---|---|
| pT217_F | 0.391 | 0.221 | 0.434 | 3.52 |
| AB42_F | 27.54 | 26.95 | 6.83 | 2.88 |
| AB40_F | 334.1 | 315.6 | 103.1 | 6.20 |
| NfL_F | 27.71 | 23.29 | 23.80 | 9.43 |
| GFAP_F | 78.75 | 63.60 | 110.6 | 18.17 |

### Correlation highlights

| Pair | Pearson r | Meaning |
|---|---|---|
| pT217_F ↔ pT217_AB42_F | **0.952** | Ratio derived from pT217_F |
| NfL_Q ↔ NfL_F | **0.934** | Same analyte, two platforms |
| GFAP_Q ↔ GFAP_F | **0.793** | Same analyte, two platforms |
| NfL_F ↔ GFAP_F | **0.770** | Both neurodegeneration markers co-elevate |
| AB42_F ↔ AB40_F | **0.742** | Both amyloid peptides from same precursor |

### Group differences (Kruskal-Wallis, 3-group)

All biomarkers significant at p<0.001. pTau-217 has the largest effect size (η²=0.350).

| Feature | H statistic | η² | Significance |
|---|---|---|---|
| pT217_F | **311.7** | **0.350** | p<0.001 |
| pT217_AB42_F | 301.6 | 0.339 | p<0.001 |
| NfL_Q | 102.6 | 0.221 | p<0.001 |
| GFAP_Q | 94.8 | 0.204 | p<0.001 |
| AB42_F | 7.8 | 0.007 | p=0.020 (weakest) |

### Regression (pTau-217 as continuous outcome)

Other biomarkers explain only R²=0.125 of pTau-217 variance — it carries largely independent
information. Tree-based models (Gradient Boosting CV R²=0.209) outperform linear models
(negative CV R²) due to heavy skewness and non-linearity.

### PCA (7 core biomarkers)

| Component | Explained Variance | Cumulative |
|---|---|---|
| PC1 | 35.5% | 35.5% |
| PC2 | 27.4% | 63.0% |
| PC3 | 15.8% | 78.7% |
| PC4+PC5 | 20.2% | 99.0% |

5 components needed to reach 99% — dataset is not highly redundant, all 7 biomarkers contribute.

---

## Reference ranges (ADNI dataset, binary subset)

| Biomarker | Control mean (p10–p90) | Dementia mean (p10–p90) | Fold change |
|---|---|---|---|
| pTau-217 (pg/mL) | 0.180 (0.071–0.345) | 0.765 (0.185–1.392) | **4.24×** |
| Amyloid-β42 (pg/mL) | 27.99 (21.16–35.20) | 26.55 (20.14–35.00) | 0.95× |
| Amyloid-β40 (pg/mL) | 324.8 (247.8–401.1) | 335.6 (254.8–428.8) | 1.03× |
| NfL (pg/mL) | 22.30 (11.54–34.47) | 37.08 (20.24–63.49) | 1.66× |
| GFAP (pg/mL) | 59.66 (30.71–98.43) | 118.2 (56.18–210.8) | **1.98×** |

---

## Classification results (full evaluation)

### Binary model — Random Forest

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | **87.4%** | [80.5%, 94.3%] |
| Balanced Accuracy | 87.1% | [79.0%, 93.8%] |
| Macro F1 | 86.9% | [78.9%, 93.4%] |
| Sensitivity (Dementia Recall) | **85.7%** | [71.9%, 96.8%] |
| Specificity (Control Recall) | 88.5% | — |
| Dementia Precision | 83.3% | — |
| ROC-AUC | **0.956** | — |
| PR-AUC | **0.935** | — |
| MCC | 0.739 | — |
| Brier Score | 0.088 | — |

Confusion matrix (n=87 test patients):

| | Predicted Control | Predicted Dementia |
|---|---|---|
| True Control | 46 (TN) | 6 (FP) |
| True Dementia | 5 (FN) | 30 (TP) |

### 3-class model — Extra Trees

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Control | 0.656 | 0.808 | 0.724 |
| MCI | 0.514 | 0.383 | 0.439 |
| Dementia | 0.629 | 0.629 | 0.629 |

MCI recall (38.3%) is the hard class — plasma biomarkers alone cannot reliably distinguish it.

### Binary bake-off (9 models, 5-fold CV)

| Model | CV Accuracy | ± Std | Dementia Recall | Selected? |
|---|---|---|---|---|
| MLP | 0.873 | ±0.071 | 0.806 | No — high variance |
| SVM | 0.866 | ±0.048 | 0.811 | No — high variance |
| Logistic Regression | 0.862 | ±0.079 | 0.811 | No — high variance |
| **Random Forest** | **0.854** | **±0.004** | **0.837** | **YES** |
| Extra Trees | 0.854 | ±0.004 | 0.832 | No |
| CatBoost | 0.854 | ±0.005 | 0.816 | No |
| LightGBM | 0.852 | ±0.007 | 0.827 | No |
| HistGradientBoosting | 0.850 | ±0.010 | 0.832 | No |
| XGBoost | 0.850 | ±0.014 | 0.806 | No |

Random Forest selected: lowest fold-to-fold variance (±0.004) + best Dementia recall (0.837)
among the low-variance cluster. MLP/SVM/LR rejected despite higher mean accuracy due to
10–20× higher variance — unreliable for a clinical tool.

---

## Feature importance

### SHAP (Binary model — top 8)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | rank_pT217_F | 0.0554 |
| 2 | log_pT217_F | 0.0306 |
| 3 | rank_NfL_harmonized | 0.0272 |
| 4 | pT217_F | 0.0252 |
| 5 | pT217_AB42_F | 0.0204 |
| 6 | log_NfL_harmonized | 0.0142 |
| 7 | pT217_GFAP_product | 0.0106 |
| 8 | NfL_harmonized | 0.0089 |

pTau-217 in three forms (raw, log, rank) accounts for ~47% of total Gini importance.
GFAP ranks last despite large univariate separation — its variance is shared with pTau/NfL.

---

## Data quality & leakage audit

### Missing values (after sentinel cleaning)

| Feature | Missing % | Reason |
|---|---|---|
| pT217_F | 0.00% | Complete |
| AB42_F / AB40_F | 0.22–0.34% | 2–3 sentinel rows |
| NfL_F / GFAP_F | **36.11%** | Platform-phase structural missingness |
| NfL_Q / GFAP_Q | **48.48%** | Platform-phase structural missingness |
| MMSE | **40.16%** | Non-random — 54% missing for Control, 18% for Dementia |

### Sentinel values found (raw data)

Documentation claimed only NfL_F/GFAP_F use −4.0. Audit found −4.0 also in AB42_F, AB40_F,
ratio columns, and an undocumented −5.0 in NfL_Q/GFAP_Q. Fix: any negative → NaN.

### Leakage audit — all 12 checks passed

| Check | Status |
|---|---|
| Target-derived features | NONE |
| MMSE in primary model | EXCLUDED (circularity) |
| PHASE/platform features | EXCLUDED (cohort shortcut) |
| Harmonization ratio | CONTROLLED (fit per fold) |
| Rank transform references | CONTROLLED (fit per fold) |
| Imputer/scaler statistics | CONTROLLED (fit per fold) |
| Train/test split timing | CONTROLLED |
| Duplicate patients | NONE (0 duplicate RIDs) |
| Calibration on test set | NONE |
| HPO on test set | NONE |
| Sentinel cleaning leakage | NONE (stateless) |
| Future information | NONE (same-visit biomarkers) |

### Phase × Group confound

χ² = 241.2, Cramér's V = **0.37**, p = 1.3×10⁻⁴⁷ — PHASE and GROUP are strongly dependent.
ADNI4 (64% of data) skews toward Control/MCI; early phases skew toward Dementia.
PHASE and platform-count features are excluded from all models.

---

## Project layout

```
config/config.yaml                  Every tunable constant — nothing hardcoded in source
data/                               Cached parquet versions of each Excel sheet
src/
  data_loader.py                    Load + validate raw data (hard-fails on integrity violations)
  preprocessing/                    Sentinel handling, imputation/scaling factory, full pipeline
  features/                         Cross-platform harmonization, ratio/log/rank engineering
  models/model_zoo.py               All 9 benchmarked models + Optuna search spaces
  evaluation/metrics.py             Full metric suite (accuracy → bootstrap CI)
  api/inference.py                  Production inference — mode="binary" or mode="three_class"
  streamlit/app.py                  Clinician UI: Predict / Model Card / Dataset Analysis tabs
  analysis/                         ← NEW: complete statistical analysis modules
    data_quality.py                 Missing values, sentinels, distributions, outlier viz
    descriptive_statistics.py       Full stats per feature and per group
    correlation.py                  Pearson / Spearman / Kendall + significance testing
    statistical_tests.py            Kruskal-Wallis, Mann-Whitney U, chi-square, effect sizes
    regression.py                   Simple + multiple regression, VIF, 6-model comparison
    classification.py               Full eval of both models — ROC, PR, threshold, calibration
    feature_analysis.py             SHAP, RF importance, PCA, pair plots, univariate/bivariate
    outlier_analysis.py             IQR, Z-score, Isolation Forest
    model_comparison.py             Final comparison table + leakage audit
notebooks/                          Numbered sequential scripts (01–12): audit → train → evaluate
tests/                              54 pytest tests — leakage safety, serialization, edge cases
artifacts/
  models/                           final_model_binary.* + final_model_biomarker_only.* + metadata
  optuna_studies_binary/            Tuned hyperparameters for all 9 models (binary task)
  optuna_studies_biomarker_only/    Tuned hyperparameters (3-class task)
reports/
  ADNI_AD_Classifier_Report.docx/.pdf   Full report on 3-class development
  BINARY_MODEL_ADDENDUM.md              Binary model rationale + results
  DATA_AUDIT_REPORT.md                  Standalone data audit
  MODEL_SELECTION.md                    Bake-off rationale
  figures/                              All generated plots (28 figures)
results/                            ← NEW: statistical analysis outputs
  statistics/                       Descriptive stats, Kruskal-Wallis, Mann-Whitney, MMSE
  correlations/                     Pearson/Spearman/Kendall matrices + heatmaps
  regression/                       Simple + multiple regression, VIF, model comparison
  classification/                   ROC, PR, confusion matrix, threshold, calibration plots
  feature_analysis/                 SHAP, RF importance, PCA, pair plots, univariate plots
  distributions/                    Missing values, boxplots, histograms, target distribution
  outliers/                         IQR, Z-score, Isolation Forest results
  model_comparison/                 Final comparison table, bake-off, leakage audit
  reports/
    ANALYSIS_REPORT.md              ← Full statistical analysis report (101 files total)
run_analysis.py                     ← NEW: single command to run complete analysis pipeline
```

---

## Reproducing results

```bash
# Full setup + launch app
python start.py

# Step by step
pip install -r requirements.txt

# Run complete statistical analysis (generates 101 files in results/)
python3 run_analysis.py

# Launch app manually
cd src/streamlit && streamlit run app.py

# Run test suite (54 tests)
cd tests && python3 -m pytest -v

# Run prediction from Python
cd src/api && python3 inference.py

# Re-run development pipeline (in order)
cd notebooks
python3 01_data_audit.py          # through 10_shap_explainability.py
python3 run_one_model_binary.py random_forest 40 100   # repeat for each of 9 models
python3 12_finalize_binary.py
```

---

## Python API

```python
from api.inference import ADNIInferencePipeline

# Binary model — recommended
pipe = ADNIInferencePipeline(mode="binary")
result = pipe.predict_one({
    "pT217_F": 0.85, "AB42_F": 26.5, "AB40_F": 355.0,
    "AB42_AB40_F": 0.075, "pT217_AB42_F": 0.032,
    "NfL_F": 45.0, "GFAP_F": 110.0, "NfL_Q": None, "GFAP_Q": None,
})
print(result.diagnosis)       # "Control" or "Dementia"
print(result.confidence)      # e.g. 0.89
print(result.model_accuracy)  # {'point_estimate': 0.8736, 'ci_95_low': 0.8046, ...}
```

---

## Key limitations

- **Binary model cannot say "MCI"** — MCI patients are forced into Control or Dementia.
  Use `mode="three_class"` if MCI flagging is needed (61.2% accuracy, MCI recall only 38.3%).
- **Concurrent diagnosis, not prognosis** — biomarkers and diagnosis are from the same visit.
  This model predicts current diagnosis, not future conversion.
- **Modest test sets** — 87 patients (binary), 134 (3-class). CIs are wide; every prediction
  output includes the interval so uncertainty is always visible.
- **No external validation cohort** — generalization beyond ADNI is unverified.
- **Phase/cohort confound** — ADNI4 (64% of data) skews toward Control/MCI; early phases
  skew toward Dementia. Phase and platform features are excluded, but residual confounding
  cannot be fully ruled out.
- **Regression assumptions violated** — biomarker distributions are heavily right-skewed and
  non-normal. Linear regression results should be interpreted cautiously.
- **MMSE excluded** — including MMSE would raise accuracy (~+10% macro-F1) but risks circular
  reasoning since MMSE is part of the diagnostic protocol.
- **Observational data** — all statistical associations are observational. Correlation ≠ causation.
- **Not a certified diagnostic device** — positioned as a triage/decision-support tool
  alongside clinical judgment.

---

## Full statistical analysis report

See [`results/reports/ANALYSIS_REPORT.md`](results/reports/ANALYSIS_REPORT.md) for the complete
report covering all 13 sections: dataset description, data quality, descriptive statistics,
correlation analysis, statistical tests, regression analysis, classification analysis,
feature importance, outlier analysis, leakage audit, model comparison, limitations, and conclusion.

**101 files generated** — 52 PNG plots + 46 CSV tables + 1 JSON summary + 1 MD report.

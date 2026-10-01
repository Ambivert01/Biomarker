# Plasma Biomarker Alzheimer's Disease Diagnostic Classifier
## Complete Technical & Clinical Documentation

---

## 1. Problem Statement

### 1.1 The Clinical Problem

Alzheimer's Disease (AD) is the most common cause of dementia, affecting millions worldwide. Early and accurate diagnosis is critical — but the gold-standard diagnostic tools (PET brain scans, cerebrospinal fluid analysis via lumbar puncture) are expensive, invasive, and not widely accessible.

**The question this project answers:** Can a machine learning model classify a patient's current diagnosis (Control / MCI / Dementia) reliably using only a blood plasma test?

This is a **concurrent diagnostic classifier** — it predicts what diagnosis a patient has *right now* based on biomarkers drawn at the same visit. It is not a prognostic model (it does not predict future conversion).

### 1.2 Why This Matters

- Blood plasma tests are minimally invasive, low-cost, and scalable
- A reliable plasma biomarker classifier could triage patients before expensive PET/CSF workups
- pTau-217 and amyloid-beta ratios in plasma have emerged as strong AD biomarkers in recent research
- The ADNI dataset provides a rare, well-curated multi-platform plasma biomarker cohort

### 1.3 The Hard Problem: Why 3-Class is Difficult

Classifying Control vs. MCI vs. Dementia from plasma biomarkers alone is a well-documented hard problem in published research. MCI (Mild Cognitive Impairment) sits at the ambiguous boundary between healthy aging and dementia — its plasma biomarker profile overlaps substantially with both Control and Dementia groups. This project achieved 61.2% accuracy on the 3-class task, consistent with published literature.

**Solution adopted:** A binary model (Control vs. Dementia only, MCI excluded) was built as the recommended production model, achieving **87.4% test accuracy** (95% CI: 80.5%–94.3%) — a clinically usable result above the 80% target.

---

## 2. Dataset

### 2.1 Source

**ADNI (Alzheimer's Disease Neuroimaging Initiative)** — a longitudinal multi-site study tracking patients across multiple visits with imaging, cognitive, and biomarker data.

- Source file: `ADNI_Diagnosis_Patients__2_.xlsx`
- Modeling table: `Final_Biomarker_Patients` sheet
- **889 patients**, one row per patient (their most recent visit with usable biomarkers)

### 2.2 Class Distribution

| Diagnosis | Code | Count | % |
|---|---|---|---|
| Control (Cognitively Normal) | 1 | 343 | 38.6% |
| MCI (Mild Cognitive Impairment) | 2 | 315 | 35.4% |
| Dementia | 3 | 231 | 26.0% |

**Binary subset** (Control + Dementia only): 574 patients → 487 dev + 87 test

### 2.3 ADNI Phases

| Phase | n | Dominant Diagnosis |
|---|---|---|
| ADNI1 | 55 | 83.6% Dementia |
| ADNI2 | 102 | 66.7% Dementia |
| ADNI3 | 161 | 24.2% Dementia |
| ADNI4 | 568 | 13.2% Dementia |
| ADNIGO | 3 | 100% Dementia |

**Critical finding:** Early ADNI phases enrolled mostly Dementia patients; ADNI4 (64% of dataset) skews toward Control/MCI. This phase-diagnosis confound is the most important data quality finding.

### 2.4 Sheet Structure

| Sheet | Grain | Rows | Purpose |
|---|---|---|---|
| Diagnosis_1/2/3 | visit-level | 6,531 / 6,586 / 3,020 | All visits per patient |
| Unique_Patients_All | 1 row/patient | 3,700 | Most recent scored visit per patient |
| Biomarkers | visit-level | 2,295 | Plasma biomarker panel |
| Final_Summary | 1 row/patient | 3,700 | Joined on RID+VISCODE (same-visit) |
| Final_Biomarker_Patients | 1 row/patient | 889 | **Modeling table** — biomarker-complete rows |

---

## 3. Medical Terms & Biomarkers

### 3.1 The Five Core Plasma Biomarkers

| Biomarker | Full Name | Platform | Medical Role |
|---|---|---|---|
| **pTau-217** | Phosphorylated Tau protein at threonine-217 | Fujirebio | Strongest single AD predictor. Tau tangles are a hallmark of AD pathology. Elevated pTau-217 indicates tau hyperphosphorylation — a direct marker of AD-type neurodegeneration. |
| **Amyloid-β42 (AB42)** | Amyloid-beta peptide, 42 amino acids | Fujirebio | Amyloid plaques are the other hallmark of AD. Low AB42 in plasma/CSF indicates amyloid is depositing in the brain (being "trapped"). |
| **Amyloid-β40 (AB40)** | Amyloid-beta peptide, 40 amino acids | Fujirebio | Denominator for the AB42/AB40 ratio. AB40 is more stable and used to normalize AB42 levels. |
| **NfL** | Neurofilament Light chain | Fujirebio + Quanterix | Marker of axonal injury / neurodegeneration. Elevated NfL indicates neurons are dying — not AD-specific but reflects disease severity. |
| **GFAP** | Glial Fibrillary Acidic Protein | Fujirebio + Quanterix | Marker of astrogliosis (brain inflammation). Elevated GFAP indicates reactive astrocytes — associated with neurodegeneration. |

### 3.2 Derived Ratio Features (Clinically Motivated)

| Feature | Formula | Clinical Meaning |
|---|---|---|
| AB42/AB40 ratio | AB42_F / AB40_F | Amyloid deposition proxy. Lower ratio = more amyloid depositing in brain. |
| pTau217/AB42 ratio | pT217_F / AB42_F | Tau-amyloid interaction. High ratio = both tau pathology AND amyloid burden. |
| GFAP/NfL ratio | GFAP_harmonized / NfL_harmonized | Neuroinflammation-to-injury ratio. Reflects astrogliosis relative to axonal damage. |
| AB40 − AB42 | AB40_F − AB42_F | Absolute amyloid deposition proxy. |
| pTau217 × GFAP | pT217_F × GFAP_harmonized / 100 | Interaction term: tau pathology combined with neuroinflammation. |

### 3.3 Reference Ranges (ADNI Binary Subset)

| Biomarker | Control Mean (p10–p90) | Dementia Mean (p10–p90) |
|---|---|---|
| pTau-217 (pg/mL) | 0.180 (0.071–0.345) | 0.765 (0.185–1.392) |
| Amyloid-β42 (pg/mL) | 27.99 (21.16–35.20) | 26.55 (20.14–35.00) |
| Amyloid-β40 (pg/mL) | 324.8 (247.8–401.1) | 335.6 (254.8–428.8) |
| NfL (pg/mL) | 22.30 (11.54–34.47) | 37.08 (20.24–63.49) |
| GFAP (pg/mL) | 59.66 (30.71–98.43) | 118.2 (56.18–210.8) |

**Key observations:**
- pTau-217 shows the largest fold-change (~4.25×) between Control and Dementia
- GFAP shows ~2× elevation in Dementia
- NfL shows ~1.66× elevation in Dementia
- AB42 and AB40 show minimal group separation (amyloid ratio is more informative than raw values)

### 3.4 MMSE (Mini-Mental State Examination)

A 30-point cognitive test used clinically to assess dementia severity. **Deliberately excluded** from the primary model because:
1. MMSE alone predicts diagnosis nearly as well as the entire biomarker panel (macro-F1 0.575 vs 0.551)
2. MMSE is part of ADNI's own diagnostic protocol — including it risks circular reasoning
3. MMSE missingness is informative of diagnosis (54% missing for Control vs. 18% for Dementia) — a clinical workflow artifact, not biology

---

## 4. Data Audit Findings

### 4.1 Critical Issues Found

**Sentinel values (BLoQ — Below Limit of Quantification):**
- Documentation claimed only NfL_F and GFAP_F use −4.0 as missing sentinel
- **Found:** AB42_F (3 rows), AB40_F (2 rows), and derived ratio columns also carry −4.0
- **Found:** NfL_Q and GFAP_Q carry −4.0 for **430 rows each**, plus an undocumented −5.0 value
- **Fix:** Any negative value in any biomarker column → treated as missing (NaN)

**MMSE missingness (non-random):**

| Group | MMSE Missing |
|---|---|
| Control | 186/343 (54.2%) |
| MCI | 127/315 (40.3%) |
| Dementia | 42/231 (18.2%) |

**Assay platform confound (most critical finding):**
- Fujirebio NfL/GFAP: only measured in ADNI4 (100%)
- Quanterix NfL/GFAP: used in ADNI1–3/ADNIGO (99–100%)
- Platform availability is a near-perfect proxy for ADNI phase
- Phase is heavily confounded with diagnosis (χ² = 241.2, p = 1.3×10⁻⁴⁷)
- **Risk:** A model using raw platform columns or PHASE as features learns cohort identity, not biology

### 4.2 Cross-Platform Agreement (138 dual-platform patients)

| Biomarker | Pearson r | Mean Ratio (Fuji/Quanterix) | Stability across groups |
|---|---|---|---|
| NfL | 0.934 | 1.465× | ±3% — stable |
| GFAP | 0.793 | 0.376× | ±6% — stable |

The stable, group-invariant ratio licenses a principled harmonization rather than arbitrary platform selection.

### 4.3 Data Integrity Checklist

| Check | Result |
|---|---|
| Duplicate rows / RID / PTID | 0 / 0 / 0 |
| Patient leakage across sheets | None — 889 unique RID in modeling table |
| Null values | Only MMSE via .isna(); all other missingness hidden in negative sentinels |
| Invalid values | Negative concentrations + MMSE=−1 (2 rows) — all recoded to NaN |
| Outliers | 1.5%–5.6% per biomarker; GFAP_F max=2,478 pg/mL retained (clinically plausible) |
| Target distribution | 38.6% / 35.4% / 26.0% — mild imbalance |

---

## 5. Methodology & Pipeline Architecture

### 5.1 Overall Flow

```
Raw Excel Data
      │
      ▼
Data Loader + Validation          ← Hard-fails on integrity violations
      │
      ▼
SentinelToNaN                     ← Stateless: any negative value → NaN
      │
      ▼
CrossPlatformHarmonizer           ← Fit on train fold only
      │  NfL_F / NfL_Q → NfL_harmonized
      │  GFAP_F / GFAP_Q → GFAP_harmonized
      ▼
ClinicalFeatureEngineer           ← Fit on train fold only (rank refs)
      │  Adds: ratios, log transforms, percentile ranks
      ▼
MMSEHandler                       ← Stateless: adds MMSE_missing flag
      │
      ▼
FeatureSetSelector                ← Config-driven column selection
      │
      ▼
ColumnTransformer                 ← Fit on train fold only
      │  Imputer (median) + Scaler (robust)
      ▼
Classifier                        ← Random Forest (binary) / Extra Trees (3-class)
      │
      ▼
CalibratedClassifierCV            ← Sigmoid calibration
      │
      ▼
Structured Prediction Output
```

**Key design principle:** Every step with a fitted statistic is a proper sklearn Transformer fit only inside `Pipeline.fit()`. This makes the entire pipeline automatically leakage-safe when used with cross_val_score, GridSearchCV, or Optuna.

### 5.2 Cross-Platform Harmonization

**Problem:** NfL and GFAP are measured on two different assay platforms (Fujirebio and Quanterix) with a systematic calibration offset between them.

**Solution:** For each training fold, compute the median ratio (Fujirebio/Quanterix) from dual-platform patients in that fold. Use this ratio to convert Quanterix readings to the Fujirebio scale, then coalesce:
- Both platforms available → average of Fujirebio and converted Quanterix
- Only one platform → use that platform (converted if Quanterix)
- Neither → NaN (only 3/889 patients)

**Leakage control:** The ratio is computed fresh inside each training fold — not once on the full dataset. This is tested explicitly in `tests/test_pipeline_integration.py`.

**Empirical ratios (from 138 dual-platform patients):**
- NfL: 1.465× (Fujirebio reads higher)
- GFAP: 0.376× (Fujirebio reads lower)

### 5.3 Feature Engineering

**19 total features in the final model (7 core + 12 engineered):**

**Core biomarkers (7):**
- pT217_F, AB42_F, AB40_F, AB42_AB40_F, pT217_AB42_F
- NfL_harmonized, GFAP_harmonized

**Interaction/ratio features (3):**
- GFAP_NfL_ratio = GFAP_harmonized / NfL_harmonized
- AB40_minus_AB42 = AB40_F − AB42_F
- pT217_GFAP_product = pT217_F × GFAP_harmonized / 100

**Log transforms (4):** log1p applied to right-skewed biomarkers (raw skewness 3.5–18.2):
- log_pT217_F, log_AB40_F, log_NfL_harmonized, log_GFAP_harmonized
- AB42_F excluded: log1p made its skew worse (−0.305)

**Percentile rank transforms (5):** Fit on training fold, applied to val/test/inference via searchsorted:
- rank_pT217_F, rank_AB42_F, rank_AB40_F, rank_NfL_harmonized, rank_GFAP_harmonized
- Robust to outliers; stable for single-row production inference

### 5.4 Feature Ablation Results

| Feature Set | # Features | Classifier | Macro F1 | Dementia Recall |
|---|---|---|---|---|
| core_only | 7 | Random Forest | 0.555 | 0.624 |
| core_only | 7 | Logistic Regression | 0.555 | 0.667 |
| core_plus_engineered | 19 | Random Forest | 0.551 | 0.628 |
| core_plus_mmse_no_engineered | 9 | Logistic Regression | 0.653 | 0.775 |
| core_plus_engineered_plus_mmse | 21 | Logistic Regression | 0.661 | 0.797 |
| everything_incl_confound_risk | 23 | Logistic Regression | 0.674 | 0.784 |

**Key findings:**
- MMSE adds ~+0.10 macro-F1 — but is excluded from primary model (circularity risk)
- Phase/platform flags give highest score (0.674) — excluded as cohort shortcut
- Engineered features help non-linear models more than linear reference models

### 5.5 Imputation Strategy Comparison

| Imputation | Scaling | Classifier | Macro F1 (mean ± std) |
|---|---|---|---|
| **Median** | **Robust** | **Logistic Regression** | **0.661 ± 0.018 (best)** |
| Median | Standard | Logistic Regression | 0.653 ± 0.016 |
| KNN (k=5) | Robust | Logistic Regression | 0.646 ± 0.013 |
| Iterative | Robust | Logistic Regression | 0.646 ± 0.020 |
| Median | Robust | Random Forest | 0.625 ± 0.015 |

**Result:** Simple median imputation + robust scaling won. After cross-platform harmonization resolves most missingness, only a handful of residual gaps remain — KNN/Iterative add noise, not information.

### 5.6 Train/Test Split Strategy

- **Test set:** 15% stratified holdout = 134 patients (3-class) / 87 patients (binary)
- **Dev pool:** 85% = 755 patients (3-class) / 487 patients (binary)
- **CV:** 5-fold stratified cross-validation on dev pool
- **Splits are patient-level** — no patient appears in both train and eval folds
- **Same random seed (42)** across both models — test sets are directly comparable

---

## 6. Model Development

### 6.1 Nine Models Benchmarked

All 9 models were hyperparameter-tuned with **Optuna** (TPE sampler, up to 40 trials, 5-fold CV, macro-F1 objective), run separately for both the biomarker-only and MMSE-augmented feature sets.

Models: Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, HistGradientBoosting, SVM, MLP

### 6.2 3-Class Bake-Off (Biomarker-Only, Primary)

| Model | CV Macro F1 | ± std | Dementia Recall | Dementia Precision |
|---|---|---|---|---|
| XGBoost | 0.5827 | 0.0226 | 0.663 | 0.653 |
| MLP | 0.5821 | 0.0540 | 0.684 | 0.635 |
| **Extra Trees** | **0.5811** | **0.0111** | **0.750** | 0.618 |
| Random Forest | 0.5797 | 0.0067 | 0.699 | 0.617 |
| SVM | 0.5748 | 0.0532 | 0.770 | 0.602 |
| CatBoost | 0.5700 | 0.0135 | 0.740 | 0.602 |
| Logistic Regression | 0.5699 | 0.0721 | 0.745 | 0.613 |
| LightGBM | 0.5622 | 0.0117 | 0.714 | 0.596 |
| HistGradientBoosting | 0.5616 | 0.0141 | 0.724 | 0.592 |

**Selection: Extra Trees** — near-top macro-F1, best Dementia recall among stable models (0.750), low variance (0.011). XGBoost/MLP/SVM rejected due to high fold-to-fold variance or low Dementia recall.

### 6.3 Binary Bake-Off (Control vs. Dementia)

| Model | CV Accuracy | ± std | Dementia Recall | Control Recall |
|---|---|---|---|---|
| MLP | 0.8726 | 0.0714 | 0.806 | 0.918 |
| SVM | 0.8664 | 0.0476 | 0.811 | 0.904 |
| Logistic Regression | 0.8623 | 0.0789 | 0.811 | 0.897 |
| **Random Forest** | **0.8541** | **0.0042** | **0.837** | 0.866 |
| Extra Trees | 0.8541 | 0.0037 | 0.832 | 0.869 |
| CatBoost | 0.8541 | 0.0046 | 0.816 | 0.880 |
| LightGBM | 0.8521 | 0.0065 | 0.827 | 0.869 |
| HistGradientBoosting | 0.8500 | 0.0101 | 0.832 | 0.863 |
| XGBoost | 0.8500 | 0.0143 | 0.806 | 0.880 |

**Selection: Random Forest** — MLP/SVM/LR rejected (std 0.05–0.08, unreliable). RF, Extra Trees, CatBoost tied at 0.8541 with std ≈ 0.004. RF selected for best Dementia recall (0.837) among the low-variance cluster.

### 6.4 Final Hyperparameters

**Binary model (Random Forest):**
- n_estimators=300, max_depth=5, min_samples_split=20, min_samples_leaf=7
- max_features="log2", class_weight="balanced_subsample"
- Imputation: median, Scaling: robust, Calibration: sigmoid

**3-class model (Extra Trees):**
- n_estimators=400, max_depth=6, min_samples_split=12, min_samples_leaf=4
- max_features=0.5, class_weight="balanced"
- Imputation: median, Scaling: robust, Calibration: sigmoid

### 6.5 Calibration

Both isotonic and sigmoid (Platt) calibration were compared via nested 5-fold CV on the dev pool using mean out-of-fold Brier score.

- **Binary model:** Sigmoid selected (Brier 0.1134 vs 0.1148 isotonic). Calibration cost almost nothing in accuracy (87.4% calibrated vs 86.2% uncalibrated).
- **3-class model:** Sigmoid selected. Tree bagging ensembles (Extra Trees) already produce smoother probabilities than boosted trees.

---

## 7. Evaluation Results

### 7.1 Binary Model — Final Test Set (n=87, evaluated once)

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

**Confusion matrix:**

| | Predicted Control | Predicted Dementia |
|---|---|---|
| True Control | 46 | 6 |
| True Dementia | 5 | 30 |

The 95% CI lower bound (80.5%) itself clears the 80% clinical target — not just the point estimate.

### 7.2 3-Class Model — Final Test Set (n=134, evaluated once)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 61.2% | [53.0%, 68.7%] |
| Balanced Accuracy | 60.6% | [52.7%, 68.3%] |
| Macro F1 | 59.7% | [51.3%, 67.2%] |
| Dementia Recall | 62.9% | [46.4%, 79.4%] |
| ROC-AUC (Dementia vs rest) | 0.857 | — |
| MCC / Cohen's κ | 0.412 / 0.407 | — |

**Per-class breakdown:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Control | 0.656 | 0.808 | 0.724 |
| MCI | 0.514 | 0.383 | 0.439 |
| Dementia | 0.629 | 0.629 | 0.629 |

MCI is the hardest class — sits at the ambiguous boundary, plasma biomarkers alone cannot reliably distinguish it.

### 7.3 Decision Threshold Operating Points (3-class model)

The default argmax rule gives Dementia recall 0.629 — below the >0.80 clinical target. A threshold sweep on the dev pool found two operating points:

| Mode | Rule | Dementia Recall | Macro F1 | MCI Recall |
|---|---|---|---|---|
| Standard argmax | argmax over probabilities | 0.629 | 0.597 | 0.383 |
| Balanced (default) | P(Dementia) ≥ 0.35 | 0.657 | 0.596 | 0.362 |
| High-sensitivity | P(Dementia) ≥ 0.225 | **0.829** | 0.564 | 0.213 |

High-sensitivity mode meets the >0.80 clinical target but flags many MCI patients as Dementia — a legitimate screening trade-off.

---

## 8. SHAP Explainability

### 8.1 Global Feature Importance (mean |SHAP value|)

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | rank_pT217_F | 0.0554 | Percentile rank of pTau-217 — most influential |
| 2 | log_pT217_F | 0.0306 | Log-transformed pTau-217 |
| 3 | rank_NfL_harmonized | 0.0272 | Percentile rank of NfL |
| 4 | pT217_F | 0.0252 | Raw pTau-217 |
| 5 | pT217_AB42_F | 0.0204 | pTau-217 / Amyloid-β42 ratio |
| 6 | log_NfL_harmonized | 0.0142 | Log NfL |
| 7 | pT217_GFAP_product | 0.0106 | pTau × GFAP interaction |
| 8 | NfL_harmonized | 0.0089 | Raw NfL |
| ... | ... | ... | ... |
| 19 | GFAP_harmonized | 0.0010 | Lowest — shared variance with pTau/NfL |

**Key insight:** pTau-217 in three forms (raw, rank, log) dominates. GFAP ranks last despite large univariate separation — because it shares variance with pTau-217 and NfL; once those are in the model, GFAP's marginal contribution shrinks.

### 8.2 Error Analysis

**False Negative Dementia (missed cases — 12 of 35 true Dementia):**

| Marker | False-Negative Mean | True-Positive Mean |
|---|---|---|
| pT217_F | 0.353 | 0.966 |
| NfL_Q | 56.2 | 31.6 |

Missed Dementia patients have ~1/3 the pTau-217 of correctly identified ones, but elevated NfL — consistent with non-amyloid/non-tau pathology (vascular or mixed etiology). The model's mean P(Dementia) for false negatives is 0.286 (genuinely uncertain) vs. 0.694 for true positives — the model is appropriately unsure, not confidently wrong.

**False Positive Dementia (14 patients):**
Most have pTau-217 elevated into the Dementia range. Two nominally-Control patients had pT217_F of 0.83 and 2.00 — consistent with preclinical/prodromal AD (biomarker positivity precedes clinical symptom onset).

---

## 9. Production Architecture

### 9.1 Two Deployed Models

| | Binary (recommended) | 3-class (research reference) |
|---|---|---|
| Task | Control vs. Dementia | Control vs. MCI vs. Dementia |
| Algorithm | Random Forest | Extra Trees |
| Test accuracy | 87.4% (CI: 80.5%–94.3%) | 61.2% |
| Test patients | 87 | 134 |
| Calibration | Sigmoid | Sigmoid |
| Features | 19 (no MMSE, no phase) | 19 (no MMSE, no phase) |

### 9.2 Inference Pipeline (`src/api/inference.py`)

```
Input dict (raw biomarker values)
    │
    ▼
1. Validation & Cleaning
   - Missing fields → NaN with warning
   - Negative values → NaN (sentinel) with warning
   - Extreme values (>1e5) → flagged but passed through
   - All missing → InputValidationError
    │
    ▼
2. Feature Engineering (fitted pipeline)
   - Harmonization, ratios, logs, ranks
    │
    ▼
3. Imputation + Scaling (fitted pipeline)
    │
    ▼
4. Prediction + Calibrated Probabilities
    │
    ▼
5. Threshold Application
   - Binary: P(Dementia) ≥ 0.5 (balanced mode)
   - 3-class: P(Dementia) ≥ 0.35 (balanced mode)
    │
    ▼
6. SHAP Explanation (TreeExplainer on uncalibrated pipeline)
    │
    ▼
7. PredictionResult (diagnosis, confidence, probabilities,
   model_accuracy with CI, top SHAP contributors, warnings)
```

### 9.3 Streamlit App — Three Tabs

**Tab 1: 🔬 Predict**
- Input: 5 core biomarkers + 2 optional derived ratios + NfL/GFAP platform variants
- Output: Diagnosis, confidence, probability bar chart, SHAP waterfall, model accuracy with CI

**Tab 2: 📋 Model Card**
- Validated test metrics, confusion matrix summary, threshold operating points
- Known limitations, scope caveats

**Tab 3: 📈 Dataset Analysis**
- Reference ranges table (Control vs. Dementia mean + p10–p90)
- Group mean ± typical range bar charts (5 biomarkers)
- Pairwise ratio comparisons (Dementia ratio vs. Control ratio for every biomarker pair)

### 9.4 Test Suite (54 pytest tests)

- `test_inference.py` — input validation, edge cases, both models
- `test_pipeline_integration.py` — leakage safety, harmonization correctness
- `test_preprocessing.py` — sentinel handling, imputation, scaling

Key assertions:
- Binary model never predicts "MCI"
- Binary model accuracy ≥ 0.80
- Every prediction includes model_accuracy block with CI
- All-missing input raises InputValidationError
- Negative values treated as sentinels

---

## 10. Use Cases

| Use Case | Model | Notes |
|---|---|---|
| Blood test triage before PET/CSF | Binary | High specificity for Control/Dementia |
| Screening for Dementia in memory clinic | Binary (high-sensitivity mode) | P(Dementia) ≥ 0.35, recall 0.857 |
| Research: MCI flagging | 3-class | 61.2% accuracy — research use only |
| Explaining individual predictions to clinicians | Both | SHAP waterfall per patient |
| Population-level biomarker analysis | Dataset Analysis tab | Reference ranges, pairwise ratios |

---

## 11. Key Limitations

1. **Binary model cannot say "MCI"** — MCI patients are forced into Control or Dementia
2. **Concurrent diagnosis, not prognosis** — same-visit biomarkers and diagnosis; does not predict future conversion
3. **Modest test sets** — 87 patients (binary), 134 (3-class); CIs are wide
4. **No external validation cohort** — generalization beyond ADNI is unverified
5. **Phase/cohort confound** — ADNI4 dominates the dataset; performance on different phase mixes is unknown
6. **Not a certified diagnostic device** — triage/decision-support tool alongside clinical judgment
7. **pTau-217 dependency** — if pTau-217 is missing, confidence drops substantially (it is the #1 SHAP feature)

---

## 12. Results Summary & Conclusion

### What was built
A production ML system with two models, a leakage-safe preprocessing pipeline, full evaluation suite, SHAP explainability, and a clinician-facing Streamlit UI — all from a single ADNI plasma biomarker dataset.

### Key numbers

| | Binary | 3-class |
|---|---|---|
| Test accuracy | **87.4%** | 61.2% |
| 95% CI | [80.5%, 94.3%] | [53.0%, 68.7%] |
| Dementia recall | 85.7% | 62.9% (82.9% high-sens) |
| ROC-AUC | 0.956 | 0.857 |
| Test patients | 87 | 134 |

### Why the binary model is recommended
The 3-class task (Control/MCI/Dementia) could not reliably clear 80% accuracy using plasma biomarkers alone — consistent with published research. Dropping MCI gives a clean, clinically-usable result. The binary model's 95% CI lower bound (80.5%) itself clears the 80% target.

### Most important technical decisions
1. **Generalized sentinel handling** — any negative value, not just −4.0 in two columns
2. **Cross-platform harmonization** — stable empirical ratio, fit per training fold (leakage-safe)
3. **Phase/platform exclusion** — prevents cohort-identity shortcuts
4. **MMSE exclusion** — prevents circular reasoning
5. **Reliability over raw accuracy** — Random Forest/Extra Trees selected for low fold-to-fold variance, not just highest mean CV score
6. **Threshold operating points** — high-sensitivity mode meets >0.80 Dementia recall clinical target

### Procedure summary
```
Data Audit → Sentinel Fix → Platform Harmonization → Feature Engineering
→ Ablation Study → Imputation Comparison → 9-Model Optuna Bake-off
→ Calibration Selection → Final Evaluation (once) → SHAP Analysis
→ Error Analysis → Binary Model (scope refinement) → Production Deployment
```

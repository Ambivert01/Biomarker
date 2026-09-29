# Plasma Biomarker Alzheimer's Disease Diagnostic Classifier

A production ML system that classifies patients as **Control** or **Dementia** from plasma
biomarker concentrations, built on the ADNI dataset. Recommended model achieves **87.4% test
accuracy** (95% CI: 80.5%–94.3%). A research-reference 3-class model (Control/MCI/Dementia,
61.2% accuracy) is also included.

## Two models, deliberately

| | **Binary (recommended)** | 3-class (research reference) |
|---|---|---|
| Predicts | Control vs. Dementia only | Control vs. MCI vs. Dementia |
| Model | Random Forest | Extra Trees |
| Test accuracy | **87.4%** (95% CI: 80.5%–94.3%) | 61.2% |
| Test patients | 87 | 134 |
| Why | The 3-class task could not reliably clear 80% accuracy using plasma biomarkers alone — consistent with published research where 3-way AD classification is a well-documented hard problem. Dropping MCI gives a clean, clinically-usable result above 80%. |

Both models are available in the app — switch via the sidebar.

## Quickstart

```bash
python start.py
```

Opens the app at `http://localhost:8501`. Windows users can double-click `start.bat`,
Mac/Linux users `start.sh`. No other setup needed — `start.py` installs all dependencies.

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
- **Pairwise ratio comparisons** — for every biomarker pair (e.g. pTau-217 ÷ NfL), two bars showing the Dementia ratio vs. the Control ratio, across all factor combinations

## Biomarkers used

| Biomarker | Platform | Role |
|---|---|---|
| pTau-217 | Fujirebio | Strongest single predictor (SHAP) |
| Amyloid-β42 | Fujirebio | Amyloid burden |
| Amyloid-β40 | Fujirebio | Amyloid burden (denominator) |
| AB42/AB40 ratio | Derived | Amyloid deposition proxy |
| pTau217/AB42 ratio | Derived | Tau-amyloid interaction |
| NfL | Fujirebio + Quanterix (harmonized) | Neurodegeneration / axonal injury |
| GFAP | Fujirebio + Quanterix (harmonized) | Neuroinflammation / astrogliosis |

NfL and GFAP are harmonized across platforms using empirically derived fixed ratios
(NfL: 1.465×, GFAP: 0.376×) computed from 138 dual-platform patients in the audit.

## Reference ranges (ADNI dataset, binary subset)

| Biomarker | Control mean (p10–p90) | Dementia mean (p10–p90) |
|---|---|---|
| pTau-217 (pg/mL) | 0.180 (0.071–0.345) | 0.765 (0.185–1.392) |
| Amyloid-β42 (pg/mL) | 27.99 (21.16–35.20) | 26.55 (20.14–35.00) |
| Amyloid-β40 (pg/mL) | 324.8 (247.8–401.1) | 335.6 (254.8–428.8) |
| NfL (pg/mL) | 22.30 (11.54–34.47) | 37.08 (20.24–63.49) |
| GFAP (pg/mL) | 59.66 (30.71–98.43) | 118.2 (56.18–210.8) |

## Project layout

```
config/config.yaml              Every tunable constant — nothing hardcoded in source
data/                           Cached parquet versions of each Excel sheet
src/
  data_loader.py                Load + validate raw data (hard-fails on integrity violations)
  preprocessing/                Sentinel handling, imputation/scaling factory, full pipeline
  features/                     Cross-platform harmonization, ratio/log/rank engineering
  models/model_zoo.py           All 9 benchmarked models + Optuna search spaces
  evaluation/metrics.py         Full metric suite (accuracy → bootstrap CI)
  api/inference.py              Production inference — mode="binary" or mode="three_class"
  streamlit/app.py              Clinician UI: Predict / Model Card / Dataset Analysis tabs
notebooks/                      Numbered sequential scripts (01–12): audit → train → evaluate
tests/                          54 pytest tests — leakage safety, serialization, edge cases
artifacts/
  models/                       final_model_binary.* + final_model_biomarker_only.* + metadata
  optuna_studies_binary/        Tuned hyperparameters for all 9 models (binary task)
  optuna_studies_biomarker_only/ Tuned hyperparameters (3-class task)
reports/
  ADNI_AD_Classifier_Report.docx/.pdf   Full report on 3-class development
  BINARY_MODEL_ADDENDUM.md              Binary model rationale + results
  DATA_AUDIT_REPORT.md                  Standalone data audit
  figures/                              All generated plots
```

## Reproducing results

```bash
# Full setup + launch app
python start.py

# Step by step
pip install -r requirements.txt

# Run test suite
cd tests && python3 -m pytest -v

# Run prediction from Python
cd src/api && python3 inference.py

# Launch app manually
cd src/streamlit && streamlit run app.py

# Re-run development pipeline (in order)
cd notebooks
python3 01_data_audit.py          # through 10_shap_explainability.py
python3 run_one_model_binary.py random_forest 40 100   # repeat for each of 9 models
python3 12_finalize_binary.py
```

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

## Key limitations

- **Binary model cannot say "MCI"** — MCI patients are forced into Control or Dementia.
  Use `mode="three_class"` if MCI flagging is needed (61.2% accuracy).
- **Concurrent diagnosis, not prognosis** — biomarkers and diagnosis are from the same visit.
- **Modest test sets** — 87 patients (binary), 134 (3-class). CIs are wide; every prediction
  output includes the interval so uncertainty is always visible.
- **No external validation cohort** — generalization beyond ADNI is unverified.
- **Not a certified diagnostic device** — positioned as a triage/decision-support tool
  alongside clinical judgment.

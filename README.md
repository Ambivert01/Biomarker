# Plasma Biomarker Alzheimer's Disease Diagnostic Classification System

Production ML system that classifies patients from ADNI plasma biomarker concentrations.
**Recommended model: Control vs. Dementia, 87.4% test accuracy.** A research-reference 3-class
model (Control/MCI/Dementia, 61.2% accuracy) is also included — see below for why both exist.
Built end-to-end: data audit → leakage-safe pipeline → 9-model Optuna benchmark (run twice, once
per model) → calibration → clinical decision thresholds → SHAP explainability → error analysis →
tested inference API → Streamlit app.

## Two models, deliberately

| | **Binary (recommended)** | 3-class (research reference) |
|---|---|---|
| Predicts | Control vs. Dementia only | Control vs. MCI vs. Dementia |
| Model | Random Forest | Extra Trees |
| Test accuracy | **87.4%** (95% CI: 80.5%–94.3%) | 61.2% |
| Why two models | The 3-class task could not reliably clear 80% accuracy using plasma biomarkers alone — this matches published research, where 3-way AD classification is a well-documented hard problem even with MRI/CSF/genetics added. Dropping MCI as a class gets a clean, reliable, clinically-usable accuracy above 80%, at the cost of no longer being able to flag MCI specifically. |

Both are wired into `src/api/inference.py` and the Streamlit app — switch with `mode="binary"` or
`mode="three_class"`. **Every prediction's output includes the model's own validated accuracy
(point estimate + 95% confidence interval)**, so you always see how trustworthy that number is
alongside the diagnosis itself.

## Quickstart (one command)

1. Install Python 3.10+ if you don't already have it: https://python.org
2. Unzip this project, open a terminal in this folder, and run:
   ```
   python start.py
   ```
   (Windows users can instead just double-click `start.bat`; Mac/Linux users can double-click `start.sh`.)
3. This installs everything needed and opens the app in your browser at `http://localhost:8501`.
4. Fill in a patient's biomarker values in the form and click **Run prediction**.

No other setup required — `start.py` handles installing dependencies for you.

**Start here:** `reports/ADNI_AD_Classifier_Report.docx` (or `.pdf`) — the full write-up of everything
below, with figures and results tables. Note: the report as written covers the original 3-class
development in depth; the binary Control-vs-Dementia model was added afterward specifically to meet
a hard 80%+ accuracy requirement — see `reports/BINARY_MODEL_ADDENDUM.md` for that part of the story.

## Project layout

```
config/config.yaml          Every tunable constant — nothing hardcoded in source
data/                        Cached parquet versions of each Excel sheet + a cleaned EDA table
src/
  data_loader.py             Load + validate raw data (hard-fails on integrity violations)
  preprocessing/              Sentinel handling, imputation/scaling factory, full pipeline assembly
  features/                   Cross-platform harmonization, ratio/log/rank engineering, MMSE handling
  models/model_zoo.py         All 9 benchmarked models + their Optuna search spaces (binary- and multiclass-safe)
  evaluation/metrics.py       Full metric suite (accuracy → bootstrap CI) as one reusable function
  api/inference.py            Production inference pipeline — mode="binary" or mode="three_class"
  streamlit/app.py            Clinician-facing UI with a model switcher + Model Card tab
notebooks/                   Numbered, sequential, reproducible analysis/training scripts (01–12)
tests/                       54 pytest tests — leakage safety, serialization, edge cases, both models
artifacts/
  models/                     final_model_binary.* (recommended) and final_model_biomarker_only.* (3-class reference), + metadata.json for each
  optuna_studies_binary/, optuna_studies_biomarker_only/, optuna_studies/   Every model's tuned hyperparameters, per experiment
reports/
  ADNI_AD_Classifier_Report.docx / .pdf   Full report on the original 3-class development
  BINARY_MODEL_ADDENDUM.md                Why + how the binary model was added, and its results
  DATA_AUDIT_REPORT.md                    Standalone data-audit writeup
  figures/                                All generated plots (EDA, evaluation, SHAP, thresholds, binary model)
  *.csv                                    Every intermediate results table (bake-offs, ablations, etc.)
```

## Reproducing / running things

```bash
# Easiest: one command does setup + launches the app
python start.py

# Or step by step:
pip install -r requirements.txt

# Run the full test suite
cd tests && python3 -m pytest -v

# Make a prediction from Python directly
cd src/api && python3 inference.py

# Launch the clinician UI manually
cd src/streamlit && streamlit run app.py

# Re-run any step of the development process (in order)
cd notebooks && python3 01_data_audit.py   # ... through 10_shap_explainability.py
# Binary model bake-off (run per-model, resumable — see file for why):
python3 run_one_model_binary.py random_forest 40 100   # repeat for each of the 9 model names
python3 12_finalize_binary.py
```

## Using a specific model from Python

```python
from api.inference import ADNIInferencePipeline

# Recommended: binary, 87.4% test accuracy
pipe = ADNIInferencePipeline(mode="binary")
result = pipe.predict_one({
    "pT217_F": 0.85, "AB42_F": 26.5, "AB40_F": 355.0,
    "AB42_AB40_F": 0.075, "pT217_AB42_F": 0.032,
    "NfL_F": 45.0, "GFAP_F": 110.0, "NfL_Q": None, "GFAP_Q": None,
})
print(result.diagnosis)          # "Control" or "Dementia"
print(result.model_accuracy)     # {'point_estimate': 0.8736, 'ci_95_low': 0.8046, ...}
```

## Key limitations (see report §11 and BINARY_MODEL_ADDENDUM.md for full discussion)

- **The binary model cannot say "MCI"** — it was trained only on Control and Dementia patients.
  Use `mode="three_class"` if you need MCI flagged, at the cost of much lower accuracy (~61%).
- **Concurrent diagnosis, not prognosis** — biomarkers and diagnosis are drawn at the same visit.
- **Modest test sets** (87 patients for binary, 134 for 3-class) — confidence intervals are wide;
  every prediction's output includes that interval so you can see the uncertainty directly.
- **No external validation cohort** — generalization beyond ADNI is unverified.

Not a certified diagnostic device; positioned as a triage / decision-support tool alongside
clinical judgment.

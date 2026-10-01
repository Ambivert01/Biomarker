# Documentation Index — ADNI Plasma Biomarker AD Classifier

All project documentation in one place. Start here.

---

## Quick Start

```bash
python start.py                          # full setup + launch app
cd src/streamlit && streamlit run app.py # launch app directly
python3 run_analysis.py                  # run statistical analysis pipeline
PYTHONPATH=src .venv/bin/pytest tests/ -v
```

---

## Documents

### Primary References

| Document | Location | What it covers |
|---|---|---|
| **README** | [`../README.md`](../README.md) | Project overview, quickstart, metrics, dataset, commands |
| **Full Technical Documentation** | [`../DOCUMENTATION.md`](../DOCUMENTATION.md) | Complete technical + clinical documentation — pipeline, methodology, evaluation, SHAP, production architecture |
| **Statistical Analysis Report** | [`../results/reports/ANALYSIS_REPORT.md`](../results/reports/ANALYSIS_REPORT.md) | All statistical findings — descriptive stats, correlations, Kruskal-Wallis, regression, classification, leakage audit |
| **Visual Guide** | [`../ADNI_Research_Analysis_Visual_Guide/ADNI_RESEARCH_ANALYSIS_VISUAL_GUIDE.md`](../ADNI_Research_Analysis_Visual_Guide/ADNI_RESEARCH_ANALYSIS_VISUAL_GUIDE.md) | Research-grade interpretation of every figure — for medical/research audience |

### Model Reports

| Document | Location | What it covers |
|---|---|---|
| **Binary Model Addendum** | [`../reports/BINARY_MODEL_ADDENDUM.md`](../reports/BINARY_MODEL_ADDENDUM.md) | Why binary model was built, bake-off results, final metrics, caveats |
| **Data Audit Report** | [`../reports/DATA_AUDIT_REPORT.md`](../reports/DATA_AUDIT_REPORT.md) | Full data integrity audit — sentinel values, platform confound, MMSE missingness |
| **Model Selection Rationale** | [`../reports/MODEL_SELECTION.md`](../reports/MODEL_SELECTION.md) | 3-class bake-off results and LightGBM vs Extra Trees decision |
| **Full Classifier Report (PDF)** | [`../reports/ADNI_AD_Classifier_Report.pdf`](../reports/ADNI_AD_Classifier_Report.pdf) | Complete formal report on 3-class model development |
| **Full Classifier Report (DOCX)** | [`../reports/ADNI_AD_Classifier_Report.docx`](../reports/ADNI_AD_Classifier_Report.docx) | Editable version of the formal report |

### Analysis Outputs

| Output | Location | What it covers |
|---|---|---|
| **Analysis Summary JSON** | [`../results/reports/analysis_summary.json`](../results/reports/analysis_summary.json) | Machine-readable summary of all analysis results |
| **Classification metrics** | [`../results/classification/`](../results/classification/) | ROC, PR, confusion matrix, threshold, calibration CSVs + PNGs |
| **Correlation matrices** | [`../results/correlations/`](../results/correlations/) | Pearson/Spearman/Kendall CSVs + heatmaps |
| **Statistical tests** | [`../results/statistics/`](../results/statistics/) | Kruskal-Wallis, Mann-Whitney, chi-square, MMSE CSVs + plots |
| **Regression results** | [`../results/regression/`](../results/regression/) | OLS coefficients, VIF, model comparison CSVs + plots |
| **Feature analysis** | [`../results/feature_analysis/`](../results/feature_analysis/) | SHAP, RF importance, PCA, univariate/bivariate plots |
| **Outlier analysis** | [`../results/outliers/`](../results/outliers/) | IQR, Z-score, Isolation Forest CSVs + plots |
| **Model comparison** | [`../results/model_comparison/`](../results/model_comparison/) | Bake-off table, leakage audit, final comparison |
| **Distributions** | [`../results/distributions/`](../results/distributions/) | Missing values, boxplots, histograms, sentinel counts |

### Original Pipeline Figures

| Folder | Location | What it covers |
|---|---|---|
| **Development figures** | [`../reports/figures/`](../reports/figures/) | 35 figures from original model development (EDA, SHAP waterfalls, ROC, calibration) |
| **Visual Guide images** | [`../ADNI_Research_Analysis_Visual_Guide/images/`](../ADNI_Research_Analysis_Visual_Guide/images/) | 57 annotated figures for GitHub/research presentation |

---

## Reading Order

**For a new reader:**
1. [`../README.md`](../README.md) — 5 min overview
2. [`../DOCUMENTATION.md`](../DOCUMENTATION.md) — full technical depth
3. [`../results/reports/ANALYSIS_REPORT.md`](../results/reports/ANALYSIS_REPORT.md) — statistical findings
4. [`../ADNI_Research_Analysis_Visual_Guide/ADNI_RESEARCH_ANALYSIS_VISUAL_GUIDE.md`](../ADNI_Research_Analysis_Visual_Guide/ADNI_RESEARCH_ANALYSIS_VISUAL_GUIDE.md) — figure-by-figure interpretation

**For a medical/clinical reader:**
1. README → Model Card section
2. Visual Guide sections 4, 5, 8, 13, 14, 15
3. DOCUMENTATION.md sections 3, 7, 10, 11

**For a code reviewer:**
1. README → Project layout
2. DOCUMENTATION.md sections 5, 6, 9
3. `src/api/inference.py`, `src/preprocessing/pipeline.py`, `src/features/`
4. `tests/` — 54 pytest tests

---

## Key Numbers (verified from CSVs)

| Metric | Value |
|---|---|
| Dataset | 889 patients, 0 duplicate RIDs |
| Binary accuracy | 87.4% (95% CI: 80.5%–94.3%) |
| ROC-AUC | 0.956 |
| PR-AUC | 0.935 |
| Brier Score | 0.088 |
| 3-class accuracy | 61.2% |
| pTau-217 Spearman r | +0.585 |
| Kruskal-Wallis H (pTau-217) | 311.7 |
| Phase confound Cramér's V | 0.37 |
| Leakage checks passed | 12/12 |
| Generated analysis files | 101 |

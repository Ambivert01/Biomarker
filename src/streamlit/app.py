"""
app.py — Streamlit interface for the ADNI plasma-biomarker AD classifier.

Run with:  streamlit run app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from api.inference import ADNIInferencePipeline, InputValidationError, THRESHOLD_MODES

# ---------------------------------------------------------------------------
# Reference statistics computed from ADNI dataset (Control=343, Dementia=231)
# Negatives treated as missing (sentinel values). p10/p90 = typical range.
# ---------------------------------------------------------------------------
BIOMARKER_STATS = {
    "pT217_F": {
        "label": "pTau-217", "unit": "pg/mL",
        "Control":  {"mean": 0.180, "p10": 0.071, "p90": 0.345},
        "Dementia": {"mean": 0.765, "p10": 0.185, "p90": 1.392},
    },
    "AB42_F": {
        "label": "Amyloid-β42", "unit": "pg/mL",
        "Control":  {"mean": 27.994, "p10": 21.158, "p90": 35.202},
        "Dementia": {"mean": 26.546, "p10": 20.144, "p90": 34.998},
    },
    "AB40_F": {
        "label": "Amyloid-β40", "unit": "pg/mL",
        "Control":  {"mean": 324.834, "p10": 247.842, "p90": 401.112},
        "Dementia": {"mean": 335.632, "p10": 254.795, "p90": 428.793},
    },
    "NfL_F": {
        "label": "NfL", "unit": "pg/mL",
        "Control":  {"mean": 22.296, "p10": 11.537, "p90": 34.468},
        "Dementia": {"mean": 37.081, "p10": 20.244, "p90": 63.492},
    },
    "GFAP_F": {
        "label": "GFAP", "unit": "pg/mL",
        "Control":  {"mean": 59.656, "p10": 30.710, "p90": 98.430},
        "Dementia": {"mean": 118.237, "p10": 56.180, "p90": 210.760},
    },
}

COLOR_CTRL = "#2E7D32"
COLOR_DEM  = "#C62828"
COLOR_PT   = "#FF6F00"  # patient marker


@st.cache_data
def load_reference_data():
    """Load ADNI binary subset (Control + Dementia) for scatter plots."""
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "Final_Biomarker_Patients.parquet"
    df = pd.read_parquet(data_path)
    df = df[df["DIAGNOSIS"].isin([1, 3])].copy()
    for c in BIOMARKER_STATS:
        if c in df.columns:
            df[c] = df[c].where(df[c] >= 0, other=float("nan"))
    df["Group"] = df["DIAGNOSIS"].map({1: "Control", 3: "Dementia"})
    return df


def plot_single_range(key: str):
    """Individual bar chart for one biomarker — same style as plot_single_pair."""
    s    = BIOMARKER_STATS[key]
    ctrl = s["Control"]
    dem  = s["Dementia"]

    fig, ax = plt.subplots(figsize=(4.8, 3.8), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    b1 = ax.bar(0, ctrl["mean"], width=0.5, color="#2a9d8f",
                edgecolor="white", linewidth=0.6, zorder=3)
    b2 = ax.bar(1, dem["mean"],  width=0.5, color="#e63946",
                edgecolor="white", linewidth=0.6, zorder=3)

    # p10-p90 range — offset right to avoid overlap
    for xpos, grp, color in [(0, ctrl, "#2a9d8f"), (1, dem, "#e63946")]:
        rx = xpos + 0.33
        ax.vlines(rx, grp["p10"], grp["p90"], color=color, linewidth=2.5, alpha=0.75, zorder=4)
        ax.hlines([grp["p10"], grp["p90"]], rx - 0.08, rx + 0.08,
                  color=color, linewidth=1.8, alpha=0.9, zorder=4)
        ax.text(rx + 0.1, grp["p90"], f"{grp['p90']:.3g}",
                va="center", ha="left", fontsize=6.5, color=color)
        ax.text(rx + 0.1, grp["p10"], f"{grp['p10']:.3g}",
                va="center", ha="left", fontsize=6.5, color=color)

    # mean labels inside bars
    for bar, val in [(b1[0], ctrl["mean"]), (b2[0], dem["mean"])]:
        ax.text(bar.get_x() + bar.get_width() / 2, val * 0.88,
                f"{val:.3g}", ha="center", va="top",
                fontsize=8, color="white", fontweight="bold", zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Control", "Dementia"], fontsize=9, color="white")
    ax.set_title(f"{s['label']}", fontsize=10, color="white", fontweight="bold", pad=6)
    ax.set_ylabel(s["unit"], fontsize=7, color="#aaaaaa")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlim(-0.6, 2.0)
    ax.set_ylim(0, max(ctrl["p90"], dem["p90"]) * 1.15)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(axis="y", color="#ffffff18", linewidth=0.5, zorder=0)
    fig.tight_layout(pad=1.0)
    return fig


def plot_single_pair(key_x: str, key_y: str):
    """Single bar chart for one biomarker pair — Dementia vs Control ratio."""
    lx = BIOMARKER_STATS[key_x]["label"]
    ly = BIOMARKER_STATS[key_y]["label"]
    dem_ratio  = BIOMARKER_STATS[key_x]["Dementia"]["mean"] / BIOMARKER_STATS[key_y]["Dementia"]["mean"]
    ctrl_ratio = BIOMARKER_STATS[key_x]["Control"]["mean"]  / BIOMARKER_STATS[key_y]["Control"]["mean"]

    fig, ax = plt.subplots(figsize=(4.0, 3.2), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    bars = ax.bar(
        ["Dementia", "Control"], [dem_ratio, ctrl_ratio],
        color=["#e63946", "#2a9d8f"], width=0.45,
        edgecolor="white", linewidth=0.6
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 0.90,
                f"{h:.3g}", ha="center", va="top",
                fontsize=8, color="white", fontweight="bold")
    ax.set_ylim(0, max(dem_ratio, ctrl_ratio) * 1.18)
    ax.set_title(f"{lx}  ÷  {ly}", fontsize=9, color="white", pad=6)
    ax.set_ylabel("Ratio", fontsize=7, color="#aaaaaa")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(axis="y", color="#ffffff18", linewidth=0.5)
    fig.tight_layout(pad=1.0)
    return fig


def plot_dataset_pairwise(ref_df: pd.DataFrame):
    """Returns list of (title, [(key_x, key_y), ...]) — rendered as st.columns grids."""
    keys = list(BIOMARKER_STATS.keys())
    groups = []
    for key_x in keys:
        others = [k for k in keys if k != key_x]
        groups.append((BIOMARKER_STATS[key_x]["label"], key_x, others))
    return groups

st.set_page_config(page_title="ADNI Plasma Biomarker AD Classifier", layout="wide", page_icon="🧠")
st.markdown("<style> h1 a, h2 a, h3 a, h4 a { display: none !important; } </style>", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline(mode):
    return ADNIInferencePipeline(mode=mode)


st.title("🧠 Plasma Biomarker Alzheimer's Diagnostic Classifier")
st.caption(
    "Research prototype — predicts diagnosis from plasma biomarkers only. "
    "Not a certified diagnostic device. See the Model Card tab for validated performance, limitations, "
    "and the clinical-scope caveats established during model development."
)

mode = st.sidebar.radio(
    "Model",
    options=["binary", "three_class"],
    format_func=lambda m: {
        "binary": "Control vs Dementia (recommended — 87.4% test accuracy)",
        "three_class": "Control vs MCI vs Dementia (research reference — 61.2% test accuracy)",
    }[m],
)
st.sidebar.caption(
    "The binary model can't say \"MCI\" — it only distinguishes Control from Dementia. "
    "The three-class model can flag MCI but is far less accurate. See report §1 for why."
)

try:
    pipeline = load_pipeline(mode)
    ref_df   = load_reference_data()
except Exception as e:
    st.error(f"Could not load the model artifacts: {e}\n\nRun notebooks/09_finalize_biomarker_only.py and "
             f"notebooks/12_finalize_binary.py first.")
    st.stop()

tab_predict, tab_model_card, tab_dataset = st.tabs(["🔬 Predict", "📋 Model Card", "📈 Dataset Analysis"])

with tab_predict:
    st.subheader("Patient Plasma Biomarker Panel")
    st.caption("Leave a field blank if that assay wasn't run — the pipeline harmonizes across "
               "Fujirebio/Quanterix platforms and tolerates a partial panel.")

    # Row 1 — Fujirebio (left) + Neuro (right)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🧪 Fujirebio Panel")
        pt217 = st.number_input("pTau-217 (pT217_F)", min_value=0.0, value=None, placeholder="e.g. 0.45", format="%.4f")
        ab42  = st.number_input("Amyloid-β42 (AB42_F)", min_value=0.0, value=None, placeholder="e.g. 27.0", format="%.3f")
        ab40  = st.number_input("Amyloid-β40 (AB40_F)", min_value=0.0, value=None, placeholder="e.g. 340.0", format="%.3f")
    with col2:
        st.markdown("### 🧠 Neurodegeneration / Neuroinflammation")
        nfl_val  = st.number_input("NfL",  min_value=0.0, value=None, placeholder="optional")
        gfap_val = st.number_input("GFAP", min_value=0.0, value=None, placeholder="optional")

    # Pass same value to both platform columns — harmonizer picks whichever is available
    nfl_f = nfl_q = nfl_val
    gfap_f = gfap_q = gfap_val

    # Row 2 — Derived ratios below, full width 2 columns
    st.markdown("### 📐 Derived Ratios *(auto-computed if left blank)*")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        ab_ratio = st.number_input("AB42/AB40 ratio",   min_value=0.0, value=None, placeholder="auto", format="%.5f")
    with col_r2:
        pt_ratio = st.number_input("pTau217/AB42 ratio", min_value=0.0, value=None, placeholder="auto", format="%.5f")

    if st.button("Run prediction", type="primary"):
        mode_label = "balanced"
        explain = True
        record = {
            "pT217_F": pt217, "AB42_F": ab42, "AB40_F": ab40,
            "AB42_AB40_F": ab_ratio if ab_ratio is not None else (ab42 / ab40 if (ab42 and ab40) else None),
            "pT217_AB42_F": pt_ratio if pt_ratio is not None else (pt217 / ab42 if (pt217 and ab42) else None),
            "NfL_F": nfl_f, "GFAP_F": gfap_f, "NfL_Q": nfl_q, "GFAP_Q": gfap_q,
        }
        try:
            result = pipeline.predict_one(record, threshold_mode=mode_label, explain=explain)
        except InputValidationError as e:
            st.error(str(e))
        else:
            color = {"Control": "green", "MCI": "orange", "Dementia": "red"}[result.diagnosis]
            st.markdown(f"## Predicted diagnosis: :{color}[{result.diagnosis}]")
            st.markdown(f"**Confidence:** {result.confidence:.1%}  \n**Decision rule:** {result.decision_rule_note}")
            acc = result.model_accuracy
            st.info(f"📊 **This model's validated accuracy:** {acc['point_estimate']:.1%} "
                    f"(95% confidence interval: {acc['ci_95_low']:.1%}–{acc['ci_95_high']:.1%}, "
                    f"measured on {acc['n_test_patients']} held-out patients never seen during training)")

            proba_df = pd.DataFrame({"Class": list(result.probabilities.keys()),
                                      "Probability": list(result.probabilities.values())})
            color_map = {"Control": "#2E7D32", "MCI": "#F9A825", "Dementia": "#C62828"}
            bar_colors = [color_map[c] for c in proba_df["Class"]]
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.barh(proba_df["Class"], proba_df["Probability"], color=bar_colors)
            ax.set_xlim(0, 1)
            for i, v in enumerate(proba_df["Probability"]):
                ax.text(v + 0.01, i, f"{v:.1%}", va='center')
            st.pyplot(fig)

            if result.warnings:
                with st.expander(f"⚠️ {len(result.warnings)} warning(s)"):
                    for w in result.warnings:
                        st.warning(w)

            if result.top_shap_contributors:
                st.subheader("Why this prediction — top contributing features")
                shap_df = pd.DataFrame(result.top_shap_contributors)
                fig2, ax2 = plt.subplots(figsize=(7, 3.5))
                colors2 = ["#C62828" if d == "increases" else "#1565C0" for d in shap_df["direction"]]
                ax2.barh(shap_df["feature"], shap_df["shap_value"], color=colors2)
                ax2.axvline(0, color='black', lw=0.8)
                ax2.set_xlabel(f"SHAP contribution toward '{result.diagnosis}'")
                ax2.invert_yaxis()
                st.pyplot(fig2)



with tab_model_card:
    st.header("Model Card")
    meta = pipeline.metadata
    scope_label = "Control vs Dementia (MCI excluded)" if mode == "binary" else "Control vs MCI vs Dementia"
    st.markdown(f"""
| | |
|---|---|
| **Model** | {meta['model_name']} ({mode}) |
| **Diagnostic scope** | {scope_label} |
| **Role** | {meta['role']} |
| **Calibration** | {meta['calibration_method']} |
| **Features used** | {len(meta['features'])} (7 core plasma biomarkers + 12 clinically-engineered ratio/log/rank features, no MMSE) |
| **Development set** | {meta['n_dev']} patients |
| **Held-out test set** | {meta['n_test']} patients (evaluated once) |
""")
    st.subheader("Test-set performance")
    tm = meta["test_metrics"]
    acc_ci = meta["bootstrap_ci_95"]["accuracy"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{tm['accuracy']:.1%}", help=f"95% CI: {acc_ci[1]:.1%}–{acc_ci[2]:.1%}")
    c2.metric("Balanced Accuracy", f"{tm['balanced_accuracy']:.3f}")
    c3.metric("Dementia Recall", f"{tm['recall_Dementia']:.3f}")
    c4.metric("Dementia Precision", f"{tm['precision_Dementia']:.3f}")

    st.subheader("Decision threshold operating points (this mode)")
    from api.inference import THRESHOLD_MODES as _TM
    st.table(pd.DataFrame(
        [{"mode": k, "P(Dementia) threshold": v} for k, v in _TM[mode].items()]
    ))

    if mode == "binary":
        st.subheader("Why this model can't say \"MCI\"")
        st.markdown("""
This model was trained only on patients diagnosed Control or Dementia — MCI patients were excluded
entirely from both training and testing. That's a deliberate scope trade-off, made because the
3-class version (Control/MCI/Dementia together) could not reliably clear 80% accuracy on this
dataset using plasma biomarkers alone — consistent with published research, where 3-way AD
classification is a well-documented hard problem. Switch to the "three_class" model in the sidebar
if you need an MCI-aware prediction instead, at the cost of much lower accuracy (~61%).
""")

    st.subheader("Known limitations (see full report for details)")
    st.markdown("""
- **Concurrent diagnosis, not prognosis.** Biomarkers and diagnosis are drawn at the same clinic visit —
  this model estimates a patient's *current* diagnostic category, not their risk of *future* conversion.
- **Cohort/assay confound.** Diagnosis prevalence varies sharply by ADNI phase, and assay platform
  availability is itself almost fully determined by phase. Phase and platform-count features are
  deliberately excluded from this model to avoid learning cohort shortcuts instead of biology.
- **Small test set** (""" + str(meta['n_test']) + """ patients) — reported confidence intervals are wide; treat point
  estimates as approximate.
- Not a substitute for clinical judgment, CSF/PET biomarkers, or a full neuropsychological work-up.
""")

with tab_dataset:
    st.header("Dataset Biomarker Analysis")
    st.caption(
        "ADNI dataset: 343 Control + 231 Dementia patients (binary subset). "
        "Negative sentinel values excluded. All values in pg/mL."
    )

    # --- Summary stats table ---
    st.markdown("#### 📊 Reference Ranges: Control vs Dementia")
    rows = []
    for k, s in BIOMARKER_STATS.items():
        rows.append({
            "Biomarker": s["label"],
            "Unit": s["unit"],
            "Control Mean": f"{s['Control']['mean']:.3g}",
            "Control p10–p90": f"{s['Control']['p10']:.3g} – {s['Control']['p90']:.3g}",
            "Dementia Mean": f"{s['Dementia']['mean']:.3g}",
            "Dementia p10–p90": f"{s['Dementia']['p10']:.3g} – {s['Dementia']['p90']:.3g}",
        })
    st.table(pd.DataFrame(rows))

    # --- Range bar chart ---
    st.markdown("#### Group Mean ± Typical Range")
    st.caption("Bars = group mean. Lines = 10th–90th percentile range.")
    keys = list(BIOMARKER_STATS.keys())
    row1 = st.columns(3)
    for col, key in zip(row1, keys[:3]):
        fig_r = plot_single_range(key)
        col.pyplot(fig_r)
        plt.close(fig_r)
    _, c1, c2, _ = st.columns([0.5, 1, 1, 0.5])
    for col, key in zip([c1, c2], keys[3:]):
        fig_r = plot_single_range(key)
        col.pyplot(fig_r)
        plt.close(fig_r)

    # --- Pairwise scatter plots ---
    st.markdown("#### 🔬 Pairwise Biomarker Comparisons")
    st.caption("Each biomarker's ratio with every other — Dementia vs Control.")
    for title, key_x, others in plot_dataset_pairwise(ref_df):
        with st.expander(f"{title} vs all", expanded=False):
            pairs = [(key_x, ky) for ky in others]
            for row_start in range(0, len(pairs), 4):
                row_pairs = pairs[row_start:row_start + 4]
                cols = st.columns(4)
                for col, (kx, ky) in zip(cols, row_pairs):
                    fig_p = plot_single_pair(kx, ky)
                    col.pyplot(fig_p)
                    plt.close(fig_p)

    # ------------------------------------------------------------------ #
    # Statistical Analysis Results (from run_analysis.py)                 #
    # ------------------------------------------------------------------ #
    RESULTS = Path(__file__).resolve().parent.parent.parent / "results"

    def _show_img(path, caption=""):
        if Path(path).exists():
            st.image(str(path), caption=caption)

    def _show_csv(path, caption=""):
        p = Path(path)
        if p.exists():
            if caption:
                st.caption(caption)
            st.dataframe(pd.read_csv(p), use_container_width=True)

    st.divider()
    st.markdown("## 📊 Statistical Analysis Results")
    st.caption("Generated by `python3 run_analysis.py`. All numbers from real ADNI data.")

    # --- Classification ---
    with st.expander("🎯 Classification Performance", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            _show_img(RESULTS/"classification/roc_curve_binary.png", "ROC Curve — Binary Model")
            _show_img(RESULTS/"classification/confusion_matrix_binary.png", "Confusion Matrix — Binary Model")
            _show_img(RESULTS/"classification/calibration_curve_binary.png", "Calibration Curve")
        with c2:
            _show_img(RESULTS/"classification/pr_curve_binary.png", "Precision-Recall Curve — Binary Model")
            _show_img(RESULTS/"classification/threshold_analysis_binary.png", "Threshold Analysis")
            _show_img(RESULTS/"classification/probability_distribution_binary.png", "Predicted Probability Distribution")
        st.markdown("**Binary model metrics (test set)**")
        _show_csv(RESULTS/"classification/binary_model_metrics.csv")
        st.markdown("**3-class model**")
        c3, c4 = st.columns(2)
        with c3:
            _show_img(RESULTS/"classification/roc_curve_three_class.png", "ROC — 3-Class Model")
        with c4:
            _show_img(RESULTS/"classification/confusion_matrix_three_class.png", "Confusion Matrix — 3-Class")
        _show_csv(RESULTS/"classification/three_class_model_metrics.csv")
        _show_img(RESULTS/"classification/model_bakeoff_comparison.png", "9-Model Bake-off Comparison")

    # --- Correlation ---
    with st.expander("🔗 Correlation Analysis"):
        c1, c2, c3 = st.columns(3)
        with c1:
            _show_img(RESULTS/"correlations/correlation_pearson_heatmap.png", "Pearson Correlation")
        with c2:
            _show_img(RESULTS/"correlations/correlation_spearman_heatmap.png", "Spearman Correlation")
        with c3:
            _show_img(RESULTS/"correlations/correlation_kendall_heatmap.png", "Kendall Correlation")
        _show_img(RESULTS/"correlations/target_feature_correlation_ranking.png", "Feature–Target Correlation Ranking")
        st.markdown("**Highly correlated pairs (|r| ≥ 0.70)**")
        _show_csv(RESULTS/"correlations/highly_correlated_pairs.csv")
        st.markdown("**Feature–target ranking (Spearman r with DIAGNOSIS)**")
        _show_csv(RESULTS/"correlations/target_feature_correlation_ranking.csv")

    # --- Statistical Tests ---
    with st.expander("🧪 Statistical Tests (Kruskal-Wallis, Mann-Whitney)"):
        c1, c2 = st.columns(2)
        with c1:
            _show_img(RESULTS/"statistics/kruskal_wallis_summary.png", "Kruskal-Wallis H & Effect Size")
        with c2:
            _show_img(RESULTS/"statistics/binary_group_comparison.png", "Cohen’s d & Fold Change (Control vs Dementia)")
        _show_csv(RESULTS/"statistics/kruskal_wallis_3group.csv", "Kruskal-Wallis results (3-group)")
        _show_csv(RESULTS/"statistics/binary_group_comparison.csv", "Binary group comparison (Control vs Dementia)")
        _show_csv(RESULTS/"statistics/chi2_phase_group.csv", "Phase × Group chi-square confound")
        c3, c4 = st.columns(2)
        with c3:
            _show_img(RESULTS/"statistics/mmse_by_group.png", "MMSE by Group")

    # --- Regression ---
    with st.expander("📈 Regression Analysis (pTau-217 as outcome)"):
        c1, c2 = st.columns(2)
        with c1:
            _show_img(RESULTS/"regression/regression_diagnostics_OLS.png", "OLS Residual Diagnostics")
        with c2:
            _show_img(RESULTS/"regression/regression_model_comparison.png", "Regression Model Comparison (5-fold CV)")
        _show_img(RESULTS/"regression/vif_multicollinearity.png", "VIF Multicollinearity")
        st.markdown("**Multiple regression summary (OLS)**")
        _show_csv(RESULTS/"regression/multiple_regression_summary.csv")
        st.markdown("**Coefficients (std error, t-stat, p-value, 95% CI)**")
        _show_csv(RESULTS/"regression/multiple_regression_coefficients.csv")
        st.markdown("**Simple linear regression (each predictor → pTau-217)**")
        _show_csv(RESULTS/"regression/simple_linear_regression.csv")
        st.markdown("**VIF multicollinearity**")
        _show_csv(RESULTS/"regression/vif_multicollinearity.csv")

    # --- Feature Analysis ---
    with st.expander("🧠 Feature Importance & PCA"):
        c1, c2 = st.columns(2)
        with c1:
            _show_img(RESULTS/"feature_analysis/shap_feature_importance.png", "SHAP Global Importance")
        with c2:
            _show_img(RESULTS/"feature_analysis/rf_feature_importance.png", "RF Gini Importance")
        _show_img(RESULTS/"feature_analysis/pca_analysis.png", "PCA: Scree + PC1/PC2 + Loadings")
        _show_csv(RESULTS/"feature_analysis/pca_explained_variance.csv", "PCA explained variance")
        _show_img(RESULTS/"feature_analysis/pairplot_core_biomarkers.png", "Pair Plot — Core Biomarkers")

    # --- Distributions & Outliers ---
    with st.expander("📊 Distributions & Outliers"):
        c1, c2 = st.columns(2)
        with c1:
            _show_img(RESULTS/"distributions/numerical_distributions.png", "Biomarker Distributions by Group")
            _show_img(RESULTS/"distributions/missing_values_bar.png", "Missing Values")
        with c2:
            _show_img(RESULTS/"distributions/boxplots_by_group.png", "Boxplots by Group")
            _show_img(RESULTS/"distributions/missing_values_heatmap.png", "Missing Value Pattern")
        _show_img(RESULTS/"outliers/outlier_summary.png", "Outlier Summary (IQR)")
        _show_img(RESULTS/"outliers/outlier_before_after_gfap.png", "GFAP_F: Before vs After Trimming")
        _show_csv(RESULTS/"outliers/outliers_iqr.csv", "IQR outlier counts per feature")

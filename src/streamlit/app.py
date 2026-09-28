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


def plot_dataset_ranges(ref_df: pd.DataFrame):
    """Bar chart: Control vs Dementia mean + p10-p90 range for all 5 biomarkers."""
    keys = list(BIOMARKER_STATS.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(3.5 * len(keys), 4))
    for ax, key in zip(axes, keys):
        s = BIOMARKER_STATS[key]
        ctrl, dem = s["Control"], s["Dementia"]
        ax.bar([0], [ctrl["mean"]], color=COLOR_CTRL, alpha=0.85, width=0.4)
        ax.bar([1], [dem["mean"]],  color=COLOR_DEM,  alpha=0.85, width=0.4)
        ax.vlines(0, ctrl["p10"], ctrl["p90"], color=COLOR_CTRL, linewidth=4, alpha=0.5)
        ax.vlines(1, dem["p10"],  dem["p90"],  color=COLOR_DEM,  linewidth=4, alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "Dementia"], fontsize=9)
        ax.set_title(f"{s['label']}\n({s['unit']})", fontsize=9)
    fig.suptitle("Dataset: Biomarker Mean ± Typical Range (p10–p90) by Group", fontsize=11)
    fig.tight_layout()
    return fig


def plot_dataset_pairwise(ref_df: pd.DataFrame):
    """
    For each pair (key_x vs key_y) — 2 bars:
      Dementia bar = mean(key_x, Dementia) / mean(key_y, Dementia)
      Control bar  = mean(key_x, Control)  / mean(key_y, Control)
    Laid out 2 subplots per row for readability.
    """
    keys = list(BIOMARKER_STATS.keys())
    figs = []
    for key_x in keys:
        others = [k for k in keys if k != key_x]
        ncols = 2
        nrows = (len(others) + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(9, 4.5 * nrows),
                                  facecolor="#1a1a2e")
        axes = np.array(axes).flatten()

        for idx, key_y in enumerate(others):
            ax = axes[idx]
            lx = BIOMARKER_STATS[key_x]["label"]
            ly = BIOMARKER_STATS[key_y]["label"]

            dem_ratio  = BIOMARKER_STATS[key_x]["Dementia"]["mean"] / BIOMARKER_STATS[key_y]["Dementia"]["mean"]
            ctrl_ratio = BIOMARKER_STATS[key_x]["Control"]["mean"]  / BIOMARKER_STATS[key_y]["Control"]["mean"]

            ax.set_facecolor("#16213e")
            bars = ax.bar(
                ["Dementia", "Control"],
                [dem_ratio, ctrl_ratio],
                color=["#e63946", "#2a9d8f"],
                width=0.45, edgecolor="white", linewidth=0.6
            )
            # value label inside bar near top — avoids overflow
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h * 0.92,
                    f"{h:.3g}",
                    ha="center", va="top",
                    fontsize=9, color="white", fontweight="bold"
                )

            ax.set_title(f"{lx}  ÷  {ly}", fontsize=10, color="white", pad=8)
            ax.set_ylabel("Ratio", fontsize=8, color="#aaaaaa")
            ax.tick_params(colors="white", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
            ax.yaxis.label.set_color("#aaaaaa")
            # add 10% top padding so label never clips
            ax.set_ylim(0, max(dem_ratio, ctrl_ratio) * 1.18)

        # hide unused subplot if odd number
        for idx in range(len(others), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            f"{BIOMARKER_STATS[key_x]['label']}  —  Dementia vs Control ratio with each biomarker",
            fontsize=12, color="white", y=1.01
        )
        fig.tight_layout(pad=2.0)
        figs.append((BIOMARKER_STATS[key_x]["label"], fig))
    return figs

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
    st.caption("Bars = group mean. Vertical lines = 10th–90th percentile range.")
    fig_r = plot_dataset_ranges(ref_df)
    st.pyplot(fig_r)
    plt.close(fig_r)

    # --- Pairwise scatter plots ---
    st.markdown("#### 🔬 Pairwise Biomarker Comparisons (full dataset)")
    st.caption("Green = Control, Red = Dementia. Each row: one biomarker vs all others.")
    for title, fig_p in plot_dataset_pairwise(ref_df):
        with st.expander(f"{title} vs all", expanded=False):
            st.pyplot(fig_p)
            plt.close(fig_p)

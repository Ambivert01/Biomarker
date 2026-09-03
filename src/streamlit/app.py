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

from api.inference import ADNIInferencePipeline, InputValidationError, THRESHOLD_MODES

st.set_page_config(page_title="ADNI Plasma Biomarker AD Classifier", layout="wide", page_icon="🧠")


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
except Exception as e:
    st.error(f"Could not load the model artifacts: {e}\n\nRun notebooks/09_finalize_biomarker_only.py and "
             f"notebooks/12_finalize_binary.py first.")
    st.stop()

tab_predict, tab_model_card = st.tabs(["🔬 Predict", "📋 Model Card"])

with tab_predict:
    st.subheader("Patient Plasma Biomarker Panel")
    st.caption("Leave a field blank if that assay wasn't run — the pipeline harmonizes across "
               "Fujirebio/Quanterix platforms and tolerates a partial panel.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Fujirebio panel**")
        pt217 = st.number_input("pTau-217 (pT217_F)", min_value=0.0, value=None, placeholder="e.g. 0.45", format="%.4f")
        ab42 = st.number_input("Amyloid-β42 (AB42_F)", min_value=0.0, value=None, placeholder="e.g. 27.0", format="%.3f")
        ab40 = st.number_input("Amyloid-β40 (AB40_F)", min_value=0.0, value=None, placeholder="e.g. 340.0", format="%.3f")
    with col2:
        st.markdown("**Derived ratios** *(auto-computed if left blank and both inputs above are given)*")
        ab_ratio = st.number_input("AB42/AB40 ratio", min_value=0.0, value=None, placeholder="auto", format="%.5f")
        pt_ratio = st.number_input("pTau217/AB42 ratio", min_value=0.0, value=None, placeholder="auto", format="%.5f")
    with col3:
        st.markdown("**Neurodegeneration / neuroinflammation**")
        nfl_f = st.number_input("NfL — Fujirebio (NfL_F)", min_value=0.0, value=None, placeholder="optional")
        gfap_f = st.number_input("GFAP — Fujirebio (GFAP_F)", min_value=0.0, value=None, placeholder="optional")
        nfl_q = st.number_input("NfL — Quanterix (NfL_Q)", min_value=0.0, value=None, placeholder="optional")
        gfap_q = st.number_input("GFAP — Quanterix (GFAP_Q)", min_value=0.0, value=None, placeholder="optional")

    st.subheader("Decision Sensitivity")
    mode_label = st.radio(
        "Operating point",
        options=["balanced", "high_sensitivity", "standard_argmax"],
        format_func=lambda m: {
            "balanced": "Balanced (recommended default)",
            "high_sensitivity": "High-sensitivity screening (maximizes Dementia recall, more false alarms)",
            "standard_argmax": "Standard argmax (no clinical re-weighting)",
        }[m],
        horizontal=True,
    )
    explain = st.checkbox("Show SHAP explanation for this prediction", value=True)

    if st.button("Run prediction", type="primary"):
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

            st.subheader("Structured response (JSON)")
            st.code(result.to_json(), language="json")

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

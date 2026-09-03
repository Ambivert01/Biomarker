"""
10_shap_explainability.py
============================
Explainability layer for the PRIMARY model (Extra Trees, biomarker-only).

Explains the base (uncalibrated) Extra Trees classifier's predictions using
SHAP TreeExplainer, applied to the fully preprocessed (post feature-
engineering, post-impute, post-scale) numeric feature matrix. Calibration
(a monotonic-ish per-class probability rescaling fit after the fact) is not
itself explained — it does not change which features drove the underlying
tree ensemble's decision, only how confident the reported probability is,
so explaining the base model is standard practice and gives the
clinically-relevant "why" behind each prediction.
"""
import sys, warnings, json
sys.path.append('../src')
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap

from data_loader import load_and_validate
from preprocessing.pipeline import get_feature_set
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
FEATURE_SET_NAME = "core_plus_engineered"
CLASS_NAMES = ["Control", "MCI", "Dementia"]

df, _ = load_and_validate("../config/config.yaml")
X_all = df.drop(columns=["DIAGNOSIS"])
y_all = df["DIAGNOSIS"]
X_dev, X_test, y_dev, y_test = train_test_split(X_all, y_all, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED)

raw_pipe = joblib.load("../artifacts/models/final_model_biomarker_only_uncalibrated.joblib")
prep = raw_pipe.named_steps["prep"]
clf = raw_pipe.named_steps["clf"]
feats = get_feature_set(FEATURE_SET_NAME)

# Transform test set (the set we explain — never fit anything here, prep is already fit on dev pool)
X_test_transformed = prep.transform(X_test)
X_test_df = pd.DataFrame(X_test_transformed, columns=feats, index=X_test.index)

X_dev_transformed = prep.transform(X_dev)
X_dev_df = pd.DataFrame(X_dev_transformed, columns=feats, index=X_dev.index)

print("Building SHAP TreeExplainer...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_test_df)   # Explanation object, shape (n_samples, n_features, n_classes)
print("SHAP values shape:", shap_values.values.shape)

# ---------------------------------------------------------------------------
# 1. Global summary (beeswarm) per class
# ---------------------------------------------------------------------------
for i, cname in enumerate(CLASS_NAMES):
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values.values[:, :, i], X_test_df, show=False, plot_size=None)
    plt.title(f"SHAP Summary — Class: {cname}")
    plt.tight_layout()
    plt.savefig(f"../reports/figures/23_shap_summary_{cname.lower()}.png", bbox_inches='tight')
    plt.close()
print("Saved per-class SHAP summary (beeswarm) plots.")

# ---------------------------------------------------------------------------
# 2. Global mean |SHAP| bar chart (overall feature importance, averaged over classes)
# ---------------------------------------------------------------------------
mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))  # mean over samples and classes
order = np.argsort(mean_abs_shap)[::-1]
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(np.array(feats)[order][:15][::-1], mean_abs_shap[order][:15][::-1], color="#1565C0")
ax.set_xlabel("Mean |SHAP value| (avg over all 3 classes)")
ax.set_title("Global Feature Importance — Primary Model (SHAP)")
plt.tight_layout(); plt.savefig("../reports/figures/24_shap_global_importance.png"); plt.close()
print("Top 10 features by mean |SHAP|:")
for f, v in zip(np.array(feats)[order][:10], mean_abs_shap[order][:10]):
    print(f"  {f:25s} {v:.4f}")

# ---------------------------------------------------------------------------
# 3. Dependence plots for the top 3 features (Dementia class)
# ---------------------------------------------------------------------------
top3 = np.array(feats)[order][:3]
for feat_name in top3:
    fig = plt.figure(figsize=(7, 5.5))
    shap.dependence_plot(feat_name, shap_values.values[:, :, 2], X_test_df, show=False, interaction_index=None)
    plt.title(f"SHAP Dependence — {feat_name} (Dementia class)")
    plt.tight_layout()
    plt.savefig(f"../reports/figures/25_shap_dependence_{feat_name}_dementia.png", bbox_inches='tight')
    plt.close()
print(f"Saved dependence plots for: {list(top3)}")

# ---------------------------------------------------------------------------
# 4. Local explanations (waterfall) for 4 representative patients:
#    a correct Dementia call, a missed Dementia (false negative), a correct
#    Control call, and a Control misclassified as Dementia (false positive)
# ---------------------------------------------------------------------------
test_pred = raw_pipe.named_steps["clf"].predict(X_test_transformed)
y_test_arr = y_test.to_numpy()

def find_example(true_label, pred_label):
    idx = np.where((y_test_arr == true_label) & (test_pred == pred_label))[0]
    return idx[0] if len(idx) else None

examples = {
    "correct_dementia": find_example(3, 3),
    "missed_dementia_predicted_MCI": find_example(3, 2),
    "missed_dementia_predicted_control": find_example(3, 1),
    "correct_control": find_example(1, 1),
    "false_positive_dementia": find_example(1, 3),
}
print("\nExample row indices found:", examples)

class_idx_map = {1: 0, 2: 1, 3: 2}
for name, idx in examples.items():
    if idx is None:
        print(f"  (no example found for {name})")
        continue
    true_label = y_test_arr[idx]
    explain_class = class_idx_map[true_label]  # explain w.r.t. the TRUE class
    fig = plt.figure(figsize=(9, 6))
    exp_single = shap.Explanation(
        values=shap_values.values[idx, :, explain_class],
        base_values=shap_values.base_values[idx, explain_class],
        data=X_test_df.iloc[idx].values,
        feature_names=feats,
    )
    shap.plots.waterfall(exp_single, show=False, max_display=12)
    plt.title(f"{name} (true={CLASS_NAMES[true_label-1]}, predicted={CLASS_NAMES[test_pred[idx]-1]})")
    plt.tight_layout()
    plt.savefig(f"../reports/figures/26_shap_waterfall_{name}.png", bbox_inches='tight')
    plt.close()
print("Saved local waterfall explanations.")

# Save numeric SHAP importances for the final report
imp_df = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)
imp_df.to_csv("../reports/shap_global_importance.csv", index=False)
print("\nDONE.")

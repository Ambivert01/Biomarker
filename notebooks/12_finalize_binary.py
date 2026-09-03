"""
12_finalize_binary.py
========================
Finalizes the BINARY Control-vs-Dementia model: Random Forest, selected from
the 9-model bake-off in reports/model_bakeoff_cv_results_binary.csv using the
same reliability (low fold-to-fold variance) + Dementia-recall priority
rationale as the primary 3-class model.
"""
import sys, os, json, warnings
sys.path.append('../src')
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_curve, precision_recall_curve, auc

from data_loader import load_and_validate
from preprocessing.pipeline import build_full_preprocessing_pipeline, get_feature_set
from models.model_zoo import build_model_with_params
from evaluation.metrics import compute_full_metrics, format_metrics_report, bootstrap_ci

RANDOM_SEED = 42
FEATURE_SET_NAME = "core_plus_engineered"
FINAL_MODEL_NAME = "random_forest"
CLASS_ORDER = [1, 3]
CLASS_NAMES = ["Control", "Dementia"]
PARAMS_PATH = f"../artifacts/optuna_studies_binary/{FINAL_MODEL_NAME}_best_params.json"

df, _ = load_and_validate("../config/config.yaml")
binary_df = df[df["DIAGNOSIS"].isin([1, 3])].copy()
X_all = binary_df.drop(columns=["DIAGNOSIS"])
y_all = binary_df["DIAGNOSIS"]

X_dev, X_test, y_dev, y_test = train_test_split(
    X_all, y_all, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED)
print(f"Dev pool: {len(X_dev)} | Test (touched now, once): {len(X_test)}")

feats = get_feature_set(FEATURE_SET_NAME)
with open(PARAMS_PATH) as f:
    best_params = json.load(f)
print("Tuned hyperparameters:", best_params)


def make_final_base_pipeline():
    prep = build_full_preprocessing_pipeline(
        numeric_features=feats, imputation_strategy="median", scaling_strategy="robust")
    clf = build_model_with_params(FINAL_MODEL_NAME, best_params, random_state=RANDOM_SEED, n_classes=2)
    return Pipeline([("prep", prep), ("clf", clf)])


print("\nComparing calibration methods (nested CV on dev pool)...")
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
calib_results = {}
for method in ["isotonic", "sigmoid"]:
    oof_proba = np.zeros(len(X_dev))
    y_dev_arr = y_dev.reset_index(drop=True)
    X_dev_reset = X_dev.reset_index(drop=True)
    for train_idx, hold_idx in outer_cv.split(X_dev_reset, y_dev_arr):
        base = make_final_base_pipeline()
        calibrated = CalibratedClassifierCV(base, method=method, cv=5)
        calibrated.fit(X_dev_reset.iloc[train_idx], y_dev_arr.iloc[train_idx])
        oof_proba[hold_idx] = calibrated.predict_proba(X_dev_reset.iloc[hold_idx])[:, 1]
    y_dev_bin = (y_dev_arr == 3).astype(int)
    brier = brier_score_loss(y_dev_bin, oof_proba)
    calib_results[method] = float(brier)
    print(f"  {method}: OOF Brier score = {brier:.4f}")

best_method = min(calib_results, key=calib_results.get)
print(f"\nSelected calibration method: {best_method}")

final_base = make_final_base_pipeline()
final_model = CalibratedClassifierCV(final_base, method=best_method, cv=5)
final_model.fit(X_dev, y_dev)
print(f"\nFinal calibrated model fit on full dev pool ({len(X_dev)} patients).")

test_pred = final_model.predict(X_test)
test_proba = final_model.predict_proba(X_test)
test_metrics = compute_full_metrics(y_test, test_pred, test_proba, class_order=CLASS_ORDER, class_names=CLASS_NAMES)
print("\n" + format_metrics_report(test_metrics, title=f"BINARY MODEL — FINAL TEST SET RESULTS ({FINAL_MODEL_NAME}, {best_method}-calibrated, n={len(X_test)})"))

ci = bootstrap_ci(y_test, test_pred, test_proba, n_iterations=1000, random_state=RANDOM_SEED)
print("\n95% Bootstrap Confidence Intervals (test set, 1000 iterations):")
for k, (mean, lo, hi) in ci.items():
    print(f"  {k:20s}: {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

raw_pipe = make_final_base_pipeline()
raw_pipe.fit(X_dev, y_dev)
raw_pred = raw_pipe.predict(X_test)
raw_proba = raw_pipe.predict_proba(X_test)
raw_metrics = compute_full_metrics(y_test, raw_pred, raw_proba, class_order=CLASS_ORDER, class_names=CLASS_NAMES)
print("\n" + format_metrics_report(raw_metrics, title="Reference: UNCALIBRATED model, same test set"))

os.makedirs("../artifacts/models", exist_ok=True)
joblib.dump(final_model, "../artifacts/models/final_model_binary.joblib")
joblib.dump(raw_pipe, "../artifacts/models/final_model_binary_uncalibrated.joblib")

with open("../artifacts/models/final_model_binary_metadata.json", "w") as f:
    json.dump({
        "model_name": FINAL_MODEL_NAME,
        "role": "BINARY (Control vs Dementia) model — MCI excluded from this framing entirely",
        "calibration_method": best_method,
        "feature_set": FEATURE_SET_NAME,
        "features": feats,
        "class_order": CLASS_ORDER, "class_names": CLASS_NAMES,
        "hyperparameters": best_params,
        "imputation_strategy": "median", "scaling_strategy": "robust",
        "random_seed": RANDOM_SEED,
        "n_dev": len(X_dev), "n_test": len(X_test),
        "test_metrics": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in test_metrics.items()},
        "bootstrap_ci_95": ci,
    }, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
import seaborn as sns
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(5.5, 5))
cm = test_metrics["confusion_matrix"]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — Binary Model, Test Set (n={len(X_test)})")
plt.tight_layout(); plt.savefig("../reports/figures/27_confusion_matrix_test_binary.png"); plt.close()

y_test_bin = (y_test == 3).astype(int)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
fpr, tpr, _ = roc_curve(y_test_bin, test_proba[:, 1])
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color="#C62828", lw=2, label=f"Dementia vs Control (AUC={roc_auc:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Binary Model, Test Set")
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig("../reports/figures/28_roc_curve_test_binary.png"); plt.close()

print("\nAll plots saved. Final BINARY model saved to artifacts/models/final_model_binary.joblib")

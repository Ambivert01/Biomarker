"""
07_finalize_and_evaluate.py
=============================
1. Rebuilds the LightGBM pipeline with its Optuna-tuned hyperparameters.
2. Compares isotonic vs. sigmoid (Platt) probability calibration via NESTED
   cross-validation on the dev pool only (test set still untouched).
3. Fits the final calibrated pipeline on the FULL dev pool.
4. Evaluates on the held-out test set EXACTLY ONCE.
5. Produces every plot/metric the brief requires and persists the final
   pipeline to artifacts/models/final_model.joblib.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize

from data_loader import load_and_validate
from preprocessing.pipeline import build_full_preprocessing_pipeline, get_feature_set
from models.model_zoo import build_model_with_params
from evaluation.metrics import compute_full_metrics, format_metrics_report, bootstrap_ci, CLASS_NAMES, CLASS_ORDER

RANDOM_SEED = 42
FEATURE_SET_NAME = "core_plus_engineered_plus_mmse"
IMPUTE_STRATEGY = "median"
SCALE_STRATEGY = "robust"
FINAL_MODEL_NAME = "lightgbm"

df, _ = load_and_validate("../config/config.yaml")
X_all = df.drop(columns=["DIAGNOSIS"])
y_all = df["DIAGNOSIS"]

X_dev, X_test, y_dev, y_test = train_test_split(
    X_all, y_all, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED)
print(f"Dev pool: {len(X_dev)} | Test (touched now, for the first and only time): {len(X_test)}")

feats = get_feature_set(FEATURE_SET_NAME)
with open(f"../artifacts/optuna_studies/{FINAL_MODEL_NAME}_best_params.json") as f:
    best_params = json.load(f)


def make_final_base_pipeline():
    prep = build_full_preprocessing_pipeline(
        numeric_features=feats, imputation_strategy=IMPUTE_STRATEGY, scaling_strategy=SCALE_STRATEGY)
    clf = build_model_with_params(FINAL_MODEL_NAME, best_params, random_state=RANDOM_SEED)
    return Pipeline([("prep", prep), ("clf", clf)])


# ---------------------------------------------------------------------------
# Calibration method comparison — nested CV, dev pool only
# ---------------------------------------------------------------------------
print("\nComparing calibration methods (nested CV on dev pool)...")
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
calib_results = {}
for method in ["isotonic", "sigmoid"]:
    oof_proba = np.zeros((len(X_dev), 3))
    y_dev_arr = y_dev.reset_index(drop=True)
    X_dev_reset = X_dev.reset_index(drop=True)
    for train_idx, hold_idx in outer_cv.split(X_dev_reset, y_dev_arr):
        base = make_final_base_pipeline()
        calibrated = CalibratedClassifierCV(base, method=method, cv=5)
        calibrated.fit(X_dev_reset.iloc[train_idx], y_dev_arr.iloc[train_idx])
        oof_proba[hold_idx] = calibrated.predict_proba(X_dev_reset.iloc[hold_idx])
    y_dev_bin = label_binarize(y_dev_arr, classes=CLASS_ORDER)
    briers = [brier_score_loss(y_dev_bin[:, i], oof_proba[:, i]) for i in range(3)]
    calib_results[method] = float(np.mean(briers))
    print(f"  {method}: mean OOF Brier score = {np.mean(briers):.4f}")

best_method = min(calib_results, key=calib_results.get)
print(f"\nSelected calibration method: {best_method} (lower OOF Brier score on dev pool)")

# ---------------------------------------------------------------------------
# Fit FINAL pipeline on the FULL dev pool
# ---------------------------------------------------------------------------
final_base = make_final_base_pipeline()
final_model = CalibratedClassifierCV(final_base, method=best_method, cv=5)
final_model.fit(X_dev, y_dev)
print("\nFinal calibrated model fit on full dev pool (755 patients).")

# ---------------------------------------------------------------------------
# EVALUATE ON TEST SET — exactly once
# ---------------------------------------------------------------------------
test_pred = final_model.predict(X_test)
test_proba = final_model.predict_proba(X_test)
test_metrics = compute_full_metrics(y_test, test_pred, test_proba)
print("\n" + format_metrics_report(test_metrics, title=f"FINAL TEST SET RESULTS ({FINAL_MODEL_NAME}, {best_method}-calibrated, n={len(X_test)})"))

ci = bootstrap_ci(y_test, test_pred, test_proba, n_iterations=1000, random_state=RANDOM_SEED)
print("\n95% Bootstrap Confidence Intervals (test set, 1000 iterations):")
for k, (mean, lo, hi) in ci.items():
    print(f"  {k:20s}: {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

# Also compare against un-calibrated (raw) predictions for reference
raw_pipe = make_final_base_pipeline()
raw_pipe.fit(X_dev, y_dev)
raw_pred = raw_pipe.predict(X_test)
raw_proba = raw_pipe.predict_proba(X_test)
raw_metrics = compute_full_metrics(y_test, raw_pred, raw_proba)
print("\n" + format_metrics_report(raw_metrics, title="Reference: UNCALIBRATED model, same test set"))

# ---------------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------------
joblib.dump(final_model, "../artifacts/models/final_model.joblib")
joblib.dump(raw_pipe, "../artifacts/models/final_model_uncalibrated.joblib")

with open("../artifacts/models/final_model_metadata.json", "w") as f:
    json.dump({
        "model_name": FINAL_MODEL_NAME,
        "calibration_method": best_method,
        "feature_set": FEATURE_SET_NAME,
        "features": feats,
        "hyperparameters": best_params,
        "imputation_strategy": IMPUTE_STRATEGY,
        "scaling_strategy": SCALE_STRATEGY,
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

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5.5))
cm = test_metrics["confusion_matrix"]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — Final Test Set (n={len(X_test)})")
plt.tight_layout(); plt.savefig("../reports/figures/12_confusion_matrix_test.png"); plt.close()

# ROC curves (OvR)
y_test_bin = label_binarize(y_test, classes=CLASS_ORDER)
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#2E7D32", "#F9A825", "#C62828"]
for i, (cname, color) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{cname} (AUC={roc_auc:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (One-vs-Rest) — Final Test Set")
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig("../reports/figures/13_roc_curves_test.png"); plt.close()

# PR curves (OvR)
fig, ax = plt.subplots(figsize=(7, 6))
for i, (cname, color) in enumerate(zip(CLASS_NAMES, colors)):
    prec, rec, _ = precision_recall_curve(y_test_bin[:, i], test_proba[:, i])
    pr_auc = auc(rec, prec)
    ax.plot(rec, prec, color=color, lw=2, label=f"{cname} (AUC={pr_auc:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves (One-vs-Rest) — Final Test Set")
ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig("../reports/figures/14_pr_curves_test.png"); plt.close()

# Calibration curve (reliability diagram) — calibrated vs uncalibrated, Dementia class
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(7, 6))
for proba, label, ls in [(test_proba, f"Calibrated ({best_method})", '-'), (raw_proba, "Uncalibrated", '--')]:
    frac_pos, mean_pred = calibration_curve((y_test == 3).astype(int), proba[:, 2], n_bins=8, strategy='quantile')
    ax.plot(mean_pred, frac_pos, marker='o', linestyle=ls, label=label)
ax.plot([0, 1], [0, 1], 'k:', label="Perfect calibration")
ax.set_xlabel("Mean predicted probability (Dementia)"); ax.set_ylabel("Observed frequency")
ax.set_title("Calibration Curve — Dementia class, Final Test Set")
ax.legend()
plt.tight_layout(); plt.savefig("../reports/figures/15_calibration_curve.png"); plt.close()

# Bootstrap CI visualization
fig, ax = plt.subplots(figsize=(8, 5))
names = list(ci.keys())
means = [ci[k][0] for k in names]
los = [ci[k][0]-ci[k][1] for k in names]
his = [ci[k][2]-ci[k][0] for k in names]
ax.errorbar(means, range(len(names)), xerr=[los, his], fmt='o', capsize=5, color="#1565C0", markersize=8)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set_xlabel("Score"); ax.set_title("Test Set Metrics — 95% Bootstrap CI (1000 iterations)")
ax.axvline(0.80, color='red', ls='--', lw=1, label='Clinical target (Dementia recall > 0.80)')
ax.legend()
plt.tight_layout(); plt.savefig("../reports/figures/16_bootstrap_ci.png"); plt.close()

print("\nAll plots saved to reports/figures/. Final model saved to artifacts/models/final_model.joblib")

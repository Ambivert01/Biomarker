"""
classification.py
==================
Comprehensive classification evaluation for BOTH existing models:
- Binary: Random Forest (Control vs Dementia, 87.4% test accuracy)
- 3-class: Extra Trees (Control/MCI/Dementia, 61.2% test accuracy)

Generates confusion matrices, ROC curves, PR curves, threshold analysis,
calibration curves, class imbalance analysis, and cross-validation results.
"""
from __future__ import annotations
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss, classification_report,
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
import joblib

logger = logging.getLogger(__name__)
RANDOM_SEED = 42

# Add src to path for existing modules
_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def run_classification_analysis(df_clean: pd.DataFrame, out_dir: Path,
                                  model_dir: Path) -> dict:
    """
    df_clean: sentinel-cleaned dataset.
    model_dir: path to artifacts/models/.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)

    results = {}

    # ── Binary model ─────────────────────────────────────────────────────────
    binary_path = model_dir / "final_model_binary.joblib"
    if binary_path.exists():
        results["binary"] = _evaluate_binary_model(df_clean, binary_path, out_dir)
    else:
        logger.warning("Binary model not found at %s", binary_path)

    # ── 3-class model ────────────────────────────────────────────────────────
    three_path = model_dir / "final_model_biomarker_only.joblib"
    if three_path.exists():
        results["three_class"] = _evaluate_three_class_model(df_clean, three_path, out_dir)
    else:
        logger.warning("3-class model not found at %s", three_path)

    # ── Class imbalance analysis ─────────────────────────────────────────────
    _analyze_class_imbalance(df_clean, out_dir)

    # ── Cross-validation comparison ──────────────────────────────────────────
    _plot_cv_bakeoff_comparison(out_dir)

    logger.info("Classification analysis complete → %s", out_dir)
    return results


def _evaluate_binary_model(df: pd.DataFrame, model_path: Path, out_dir: Path) -> dict:
    model = joblib.load(model_path)
    df_bin = df[df["DIAGNOSIS"].isin([1, 3])].copy()
    X = df_bin.drop(columns=["DIAGNOSIS"])
    y = df_bin["DIAGNOSIS"].values

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    class_order = [1, 3]
    class_names = ["Control", "Dementia"]

    metrics = _compute_binary_metrics(y_test, y_pred, y_proba, class_order, class_names)
    pd.DataFrame([metrics]).to_csv(out_dir / "binary_model_metrics.csv", index=False)

    _plot_confusion_matrix(y_test, y_pred, class_names, "Binary Model", out_dir, "binary")
    _plot_roc_curve_binary(y_test, y_proba[:, 1], "Binary Model (RF)", out_dir, "binary")
    _plot_pr_curve_binary(y_test, y_proba[:, 1], "Binary Model (RF)", out_dir, "binary")
    _plot_threshold_analysis(y_test, y_proba[:, 1], out_dir, "binary")
    _plot_calibration(y_test, y_proba[:, 1], "Binary Model", out_dir, "binary")
    _plot_probability_distribution(y_test, y_proba[:, 1], class_names, out_dir, "binary")

    # Cross-validation on dev pool
    cv_results = _run_cv_evaluation(model, X_dev, y_dev, class_order, class_names, out_dir, "binary")
    metrics["cv_results"] = cv_results

    logger.info("Binary model: accuracy=%.3f, ROC-AUC=%.3f", metrics["accuracy"], metrics["roc_auc"])
    return metrics


def _evaluate_three_class_model(df: pd.DataFrame, model_path: Path, out_dir: Path) -> dict:
    model = joblib.load(model_path)
    X = df.drop(columns=["DIAGNOSIS"])
    y = df["DIAGNOSIS"].values
    class_order = [1, 2, 3]
    class_names = ["Control", "MCI", "Dementia"]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = _compute_multiclass_metrics(y_test, y_pred, y_proba, class_order, class_names)
    pd.DataFrame([{k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}]).to_csv(
        out_dir / "three_class_model_metrics.csv", index=False)

    _plot_confusion_matrix(y_test, y_pred, class_names, "3-Class Model", out_dir, "three_class")
    _plot_roc_multiclass(y_test, y_proba, class_order, class_names, out_dir)
    _plot_pr_multiclass(y_test, y_proba, class_order, class_names, out_dir)
    _plot_threshold_analysis_multiclass(y_test, y_proba, class_order, out_dir)

    logger.info("3-class model: accuracy=%.3f, macro-F1=%.3f", metrics["accuracy"], metrics["macro_f1"])
    return metrics


def _compute_binary_metrics(y_true, y_pred, y_proba, class_order, class_names) -> dict:
    dem_idx = class_order.index(3)
    y_bin = (y_true == 3).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "precision_dementia": round(float(precision_score(y_true, y_pred, pos_label=3)), 4),
        "recall_dementia": round(float(recall_score(y_true, y_pred, pos_label=3)), 4),
        "f1_dementia": round(float(f1_score(y_true, y_pred, pos_label=3)), 4),
        "precision_control": round(float(precision_score(y_true, y_pred, pos_label=1)), 4),
        "recall_control": round(float(recall_score(y_true, y_pred, pos_label=1)), 4),
        "specificity": round(float(recall_score(y_true, y_pred, pos_label=1)), 4),
        "sensitivity": round(float(recall_score(y_true, y_pred, pos_label=3)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_bin, y_proba[:, dem_idx])), 4),
        "pr_auc": round(float(average_precision_score(y_bin, y_proba[:, dem_idx])), 4),
        "brier_score": round(float(brier_score_loss(y_bin, y_proba[:, dem_idx])), 4),
        "n_test": int(len(y_true)),
        "n_dementia": int((y_true == 3).sum()),
        "n_control": int((y_true == 1).sum()),
    }


def _compute_multiclass_metrics(y_true, y_pred, y_proba, class_order, class_names) -> dict:
    y_bin = label_binarize(y_true, classes=class_order)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "roc_auc_macro": round(float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")), 4),
        "n_test": int(len(y_true)),
    }
    for i, cname in enumerate(class_names):
        metrics[f"precision_{cname}"] = round(float(precision_score(y_true, y_pred, labels=class_order, average=None)[i]), 4)
        metrics[f"recall_{cname}"] = round(float(recall_score(y_true, y_pred, labels=class_order, average=None)[i]), 4)
        metrics[f"f1_{cname}"] = round(float(f1_score(y_true, y_pred, labels=class_order, average=None)[i]), 4)
    return metrics


def _run_cv_evaluation(model, X, y, class_order, class_names, out_dir, tag) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    y_pred_cv = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
    cv_metrics = {
        "cv_accuracy": round(float(accuracy_score(y, y_pred_cv)), 4),
        "cv_macro_f1": round(float(f1_score(y, y_pred_cv, average="macro")), 4),
        "cv_balanced_accuracy": round(float(balanced_accuracy_score(y, y_pred_cv)), 4),
    }
    pd.DataFrame([cv_metrics]).to_csv(out_dir / f"{tag}_cv_metrics.csv", index=False)
    return cv_metrics


def _plot_confusion_matrix(y_true, y_pred, class_names, title, out_dir, tag):
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0], cbar=False)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Confusion Matrix — {title}")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1], cbar=False)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"Normalized Confusion Matrix — {title}")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_matrix_{tag}.png", dpi=150)
    plt.close()


def _plot_roc_curve_binary(y_true, y_score, title, out_dir, tag):
    y_bin = (y_true == 3).astype(int)
    fpr, tpr, thresholds = roc_curve(y_bin, y_score)
    auc = roc_auc_score(y_bin, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#e63946", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e63946")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curve — {title}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"roc_curve_{tag}.png", dpi=150)
    plt.close()


def _plot_pr_curve_binary(y_true, y_score, title, out_dir, tag):
    y_bin = (y_true == 3).astype(int)
    precision, recall, _ = precision_recall_curve(y_bin, y_score)
    ap = average_precision_score(y_bin, y_score)
    baseline = y_bin.mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#457b9d", lw=2, label=f"PR curve (AP = {ap:.3f})")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline (prevalence={baseline:.2f})")
    ax.fill_between(recall, precision, alpha=0.1, color="#457b9d")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {title}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"pr_curve_{tag}.png", dpi=150)
    plt.close()


def _plot_threshold_analysis(y_true, y_score, out_dir, tag):
    y_bin = (y_true == 3).astype(int)
    thresholds = np.linspace(0.1, 0.9, 81)
    rows = []
    for t in thresholds:
        y_pred_t = (y_score >= t).astype(int)
        if y_pred_t.sum() == 0 or y_pred_t.sum() == len(y_pred_t):
            continue
        rows.append({
            "threshold": round(float(t), 3),
            "precision": float(precision_score(y_bin, y_pred_t, zero_division=0)),
            "recall": float(recall_score(y_bin, y_pred_t, zero_division=0)),
            "f1": float(f1_score(y_bin, y_pred_t, zero_division=0)),
            "specificity": float(recall_score(1 - y_bin, 1 - y_pred_t, zero_division=0)),
            "accuracy": float(accuracy_score(y_bin, y_pred_t)),
        })
    thresh_df = pd.DataFrame(rows)
    thresh_df.to_csv(out_dir / f"threshold_analysis_{tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresh_df["threshold"], thresh_df["precision"], label="Precision", color="#457b9d", lw=2)
    ax.plot(thresh_df["threshold"], thresh_df["recall"], label="Recall (Sensitivity)", color="#e63946", lw=2)
    ax.plot(thresh_df["threshold"], thresh_df["f1"], label="F1-score", color="#2a9d8f", lw=2)
    ax.plot(thresh_df["threshold"], thresh_df["specificity"], label="Specificity", color="#e9c46a", lw=2)
    ax.axvline(0.5, color="gray", linestyle="--", lw=1, label="Default threshold (0.5)")
    ax.axvline(0.35, color="purple", linestyle=":", lw=1.5, label="High-sensitivity threshold (0.35)")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis — Binary Model (Dementia class)")
    ax.legend(loc="center left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"threshold_analysis_{tag}.png", dpi=150)
    plt.close()


def _plot_calibration(y_true, y_score, title, out_dir, tag):
    y_bin = (y_true == 3).astype(int)
    fraction_pos, mean_pred = calibration_curve(y_bin, y_score, n_bins=10)
    brier = brier_score_loss(y_bin, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(mean_pred, fraction_pos, "s-", color="#e63946", lw=2, label=f"Model (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Dementia)")
    ax.set_title(f"Calibration Curve — {title}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"calibration_curve_{tag}.png", dpi=150)
    plt.close()


def _plot_probability_distribution(y_true, y_score, class_names, out_dir, tag):
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color, name in [(1, "#2a9d8f", "Control"), (3, "#e63946", "Dementia")]:
        mask = y_true == label
        ax.hist(y_score[mask], bins=30, alpha=0.6, color=color, label=name, density=True)
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold=0.5")
    ax.axvline(0.35, color="purple", linestyle=":", lw=1.5, label="Threshold=0.35")
    ax.set_xlabel("P(Dementia)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted Probability Distribution by True Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"probability_distribution_{tag}.png", dpi=150)
    plt.close()


def _plot_roc_multiclass(y_true, y_proba, class_order, class_names, out_dir):
    y_bin = label_binarize(y_true, classes=class_order)
    colors = ["#2a9d8f", "#e9c46a", "#e63946"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cname, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cname} vs Rest (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — 3-Class Model (One-vs-Rest)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve_three_class.png", dpi=150)
    plt.close()


def _plot_pr_multiclass(y_true, y_proba, class_order, class_names, out_dir):
    y_bin = label_binarize(y_true, classes=class_order)
    colors = ["#2a9d8f", "#e9c46a", "#e63946"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cname, color) in enumerate(zip(class_names, colors)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        ax.plot(rec, prec, color=color, lw=2, label=f"{cname} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — 3-Class Model")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve_three_class.png", dpi=150)
    plt.close()


def _plot_threshold_analysis_multiclass(y_true, y_proba, class_order, out_dir):
    dem_idx = class_order.index(3)
    y_dem_bin = (y_true == 3).astype(int)
    thresholds = np.linspace(0.1, 0.8, 71)
    rows = []
    for t in thresholds:
        y_pred_t = np.where(
            y_proba[:, dem_idx] >= t,
            3,
            np.array(class_order)[np.argmax(
                np.delete(y_proba, dem_idx, axis=1), axis=1)])
        rows.append({
            "threshold": round(float(t), 3),
            "dementia_recall": float(recall_score(y_true, y_pred_t, labels=[3], average="macro", zero_division=0)),
            "dementia_precision": float(precision_score(y_true, y_pred_t, labels=[3], average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred_t, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred_t)),
        })
    thresh_df = pd.DataFrame(rows)
    thresh_df.to_csv(out_dir / "threshold_analysis_three_class.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresh_df["threshold"], thresh_df["dementia_recall"], label="Dementia Recall", color="#e63946", lw=2)
    ax.plot(thresh_df["threshold"], thresh_df["dementia_precision"], label="Dementia Precision", color="#457b9d", lw=2)
    ax.plot(thresh_df["threshold"], thresh_df["macro_f1"], label="Macro F1", color="#2a9d8f", lw=2)
    ax.axvline(0.35, color="purple", linestyle=":", lw=1.5, label="Balanced threshold (0.35)")
    ax.axvline(0.225, color="orange", linestyle=":", lw=1.5, label="High-sensitivity (0.225)")
    ax.set_xlabel("P(Dementia) Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis — 3-Class Model (Dementia class)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_analysis_three_class.png", dpi=150)
    plt.close()


def _analyze_class_imbalance(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 3-class
    vc3 = df["GROUP"].value_counts()
    colors = ["#2a9d8f", "#e9c46a", "#e63946"]
    axes[0].pie(vc3.values, labels=vc3.index, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops=dict(edgecolor="white"))
    axes[0].set_title("3-Class Distribution\n(Control/MCI/Dementia)")

    # Binary
    df_bin = df[df["DIAGNOSIS"].isin([1, 3])]
    vc2 = df_bin["GROUP"].value_counts()
    axes[1].pie(vc2.values, labels=vc2.index, colors=["#2a9d8f", "#e63946"],
                autopct="%1.1f%%", startangle=90, wedgeprops=dict(edgecolor="white"))
    axes[1].set_title("Binary Distribution\n(Control vs Dementia)")
    plt.suptitle("Class Distribution Analysis", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "class_imbalance.png", dpi=150)
    plt.close()

    # Imbalance report
    imbalance_rows = []
    for label, name in [(1, "Control"), (2, "MCI"), (3, "Dementia")]:
        n = int((df["DIAGNOSIS"] == label).sum())
        imbalance_rows.append({
            "class": name, "code": label, "count": n,
            "percentage": round(100 * n / len(df), 2),
            "imbalance_ratio_vs_majority": round(vc3.max() / n, 3),
        })
    pd.DataFrame(imbalance_rows).to_csv(out_dir / "class_imbalance_report.csv", index=False)


def _plot_cv_bakeoff_comparison(out_dir: Path):
    """Plot the existing bake-off results from reports/."""
    csv_path = Path(__file__).resolve().parent.parent.parent / "reports" / "model_bakeoff_cv_results_binary.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path).sort_values("cv_accuracy_mean", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#e63946" if m == "random_forest" else "#457b9d" for m in df["model"]]
    axes[0].barh(df["model"], df["cv_accuracy_mean"], color=colors, edgecolor="white")
    axes[0].errorbar(df["cv_accuracy_mean"], range(len(df)),
                     xerr=df["cv_accuracy_std"], fmt="none", color="black", capsize=3)
    axes[0].set_xlabel("CV Accuracy (mean ± std)")
    axes[0].set_title("Binary Model Bake-off: CV Accuracy\n(Red = selected model)")

    axes[1].barh(df["model"], df["cv_dementia_recall"], color=colors, edgecolor="white")
    axes[1].set_xlabel("CV Dementia Recall")
    axes[1].set_title("Binary Model Bake-off: Dementia Recall")
    plt.tight_layout()
    plt.savefig(out_dir / "model_bakeoff_comparison.png", dpi=150)
    plt.close()

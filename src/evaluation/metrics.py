"""
metrics.py
===========
One function that computes every metric the project brief asks for, given
true labels, predicted labels, and predicted probabilities. Used
identically for CV screening, HPO objectives, and the final test-set
report, so numbers are never computed two different ways in two different
places.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score,
    brier_score_loss, classification_report, precision_recall_curve, roc_curve,
)
from sklearn.preprocessing import label_binarize

CLASS_ORDER = [1, 2, 3]
CLASS_NAMES = ["Control", "MCI", "Dementia"]


def compute_full_metrics(y_true, y_pred, y_proba=None, class_order=CLASS_ORDER, class_names=CLASS_NAMES) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}

    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    out["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    out["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")
    out["mcc"] = matthews_corrcoef(y_true, y_pred)
    out["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    per_class_precision = precision_score(y_true, y_pred, labels=class_order, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=class_order, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=class_order, average=None, zero_division=0)
    for i, cname in enumerate(class_names):
        out[f"precision_{cname}"] = per_class_precision[i]
        out[f"recall_{cname}"] = per_class_recall[i]
        out[f"f1_{cname}"] = per_class_f1[i]

    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=class_order)

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        y_true_bin = label_binarize(y_true, classes=class_order)
        if len(class_order) == 2:
            # sklearn's label_binarize collapses binary problems to a single
            # column (membership in class_order[1]) instead of one column per
            # class, unlike the 3+-class case. Expand it back to one column
            # per class so the rest of this function (written for the general
            # multi-class case) doesn't need two separate code paths.
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        try:
            out["roc_auc_ovr_macro"] = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
            out["roc_auc_ovr_weighted"] = roc_auc_score(y_true_bin, y_proba, average="weighted", multi_class="ovr")
            for i, cname in enumerate(class_names):
                out[f"roc_auc_{cname}_vs_rest"] = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        except ValueError as e:
            out["roc_auc_error"] = str(e)

        # Brier score (multi-class = mean of one-vs-rest Brier scores)
        briers = []
        for i in range(len(class_order)):
            briers.append(brier_score_loss(y_true_bin[:, i], y_proba[:, i]))
        out["brier_score_mean"] = float(np.mean(briers))
        for i, cname in enumerate(class_names):
            out[f"brier_{cname}"] = briers[i]

    return out


def bootstrap_ci(y_true, y_pred, y_proba=None, n_iterations=1000, alpha=0.05, random_state=42):
    """95% CI (default) for accuracy, macro-F1, balanced accuracy, and Dementia recall via bootstrap resampling."""
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    metrics_boot = {"accuracy": [], "macro_f1": [], "balanced_accuracy": [], "recall_Dementia": []}
    for _ in range(n_iterations):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        metrics_boot["accuracy"].append(accuracy_score(yt, yp))
        metrics_boot["macro_f1"].append(f1_score(yt, yp, average="macro"))
        metrics_boot["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        metrics_boot["recall_Dementia"].append(recall_score(yt, yp, labels=[3], average="macro", zero_division=0))
    ci = {}
    for k, v in metrics_boot.items():
        v = np.array(v)
        lo, hi = np.percentile(v, [100*alpha/2, 100*(1-alpha/2)])
        ci[k] = (float(np.mean(v)), float(lo), float(hi))
    return ci


def format_metrics_report(metrics: dict, title: str = "", class_names=None) -> str:
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    lines.append(f"Accuracy:            {metrics['accuracy']:.4f}")
    lines.append(f"Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}")
    lines.append(f"Macro F1:            {metrics['macro_f1']:.4f}")
    lines.append(f"Weighted F1:         {metrics['weighted_f1']:.4f}")
    lines.append(f"Matthews Corr Coef:  {metrics['mcc']:.4f}")
    lines.append(f"Cohen's Kappa:       {metrics['cohen_kappa']:.4f}")
    if "roc_auc_ovr_macro" in metrics:
        lines.append(f"ROC-AUC (OvR macro): {metrics['roc_auc_ovr_macro']:.4f}")
        lines.append(f"Brier score (mean):  {metrics['brier_score_mean']:.4f}")
    lines.append("")
    lines.append(f"{'Class':<10}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    # Derive class names from whichever precision_* keys are actually present,
    # rather than hardcoding the 3-class (Control/MCI/Dementia) default --
    # this function is shared by the 3-class AND the binary Control-vs-Dementia model.
    if class_names is None:
        class_names = [k[len("precision_"):] for k in metrics if k.startswith("precision_")]
    for cname in class_names:
        lines.append(f"{cname:<10}{metrics[f'precision_{cname}']:>10.4f}{metrics[f'recall_{cname}']:>10.4f}{metrics[f'f1_{cname}']:>10.4f}")
    lines.append("")
    lines.append(f"Confusion Matrix (rows=true, cols=pred) {class_names}:")
    lines.append(str(metrics["confusion_matrix"]))
    return "\n".join(lines)

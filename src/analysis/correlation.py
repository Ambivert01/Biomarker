"""
correlation.py
===============
Pearson, Spearman, and Kendall correlation analysis with significance testing.
Generates heatmaps, ranked tables, and highly-correlated feature pairs.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from itertools import combinations

logger = logging.getLogger(__name__)

BIO_COLS = [
    "pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F",
    "NfL_Q", "GFAP_Q", "NfL_F", "GFAP_F",
]
MMSE_COL = "MMSE_FINAL_SCORE"


def run_correlation_analysis(df: pd.DataFrame, out_dir: Path) -> dict:
    """
    df must already have sentinels cleaned.
    Computes Pearson, Spearman, Kendall correlations + significance.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_cols = [c for c in BIO_COLS + [MMSE_COL] if c in df.columns]
    # Add numeric target for target-correlation ranking
    df = df.copy()
    df["DIAGNOSIS_num"] = df["DIAGNOSIS"]  # 1=Control, 2=MCI, 3=Dementia

    analysis_cols = num_cols + ["DIAGNOSIS_num"]
    df_num = df[analysis_cols].dropna(how="all")

    # ── 1. Correlation matrices ─────────────────────────────────────────────
    results = {}
    for method in ["pearson", "spearman", "kendall"]:
        corr_mat = df_num.corr(method=method, numeric_only=True)
        corr_mat.to_csv(out_dir / f"correlation_{method}.csv")
        results[f"{method}_matrix"] = corr_mat.to_dict()
        _plot_heatmap(corr_mat, method, out_dir)

    # ── 2. Pairwise significance (Pearson + Spearman) ───────────────────────
    for method in ["pearson", "spearman"]:
        sig_rows = _pairwise_significance(df_num, analysis_cols, method)
        sig_df = pd.DataFrame(sig_rows)
        sig_df.to_csv(out_dir / f"correlation_{method}_significance.csv", index=False)

    # ── 3. Target-vs-feature ranking ────────────────────────────────────────
    target_rows = []
    for col in num_cols:
        sub = df[["DIAGNOSIS_num", col]].dropna()
        if len(sub) < 10:
            continue
        r_p, p_p = stats.pearsonr(sub["DIAGNOSIS_num"], sub[col])
        r_s, p_s = stats.spearmanr(sub["DIAGNOSIS_num"], sub[col])
        target_rows.append({
            "feature": col,
            "pearson_r": round(float(r_p), 4),
            "pearson_p": round(float(p_p), 6),
            "spearman_r": round(float(r_s), 4),
            "spearman_p": round(float(p_s), 6),
            "abs_pearson_r": round(abs(float(r_p)), 4),
            "abs_spearman_r": round(abs(float(r_s)), 4),
            "n": int(len(sub)),
        })
    target_df = pd.DataFrame(target_rows).sort_values("abs_spearman_r", ascending=False)
    target_df.to_csv(out_dir / "target_feature_correlation_ranking.csv", index=False)
    _plot_target_correlation(target_df, out_dir)

    # ── 4. Highly correlated feature pairs ──────────────────────────────────
    pearson_mat = df_num[num_cols].corr(method="pearson")
    high_corr = []
    for c1, c2 in combinations(num_cols, 2):
        r = pearson_mat.loc[c1, c2]
        if abs(r) >= 0.70:
            high_corr.append({"feature_1": c1, "feature_2": c2,
                               "pearson_r": round(float(r), 4),
                               "abs_r": round(abs(float(r)), 4)})
    high_corr_df = pd.DataFrame(high_corr).sort_values("abs_r", ascending=False)
    high_corr_df.to_csv(out_dir / "highly_correlated_pairs.csv", index=False)

    # ── 5. Binary-subset correlation (Control vs Dementia only) ─────────────
    df_bin = df[df["DIAGNOSIS"].isin([1, 3])].copy()
    df_bin["binary_target"] = (df_bin["DIAGNOSIS"] == 3).astype(int)
    bin_rows = []
    for col in num_cols:
        sub = df_bin[["binary_target", col]].dropna()
        if len(sub) < 10:
            continue
        r_p, p_p = stats.pearsonr(sub["binary_target"], sub[col])
        r_s, p_s = stats.spearmanr(sub["binary_target"], sub[col])
        bin_rows.append({
            "feature": col,
            "pearson_r": round(float(r_p), 4),
            "pearson_p": round(float(p_p), 6),
            "spearman_r": round(float(r_s), 4),
            "spearman_p": round(float(p_s), 6),
            "n": int(len(sub)),
        })
    bin_df = pd.DataFrame(bin_rows)
    bin_df["abs_spearman_r"] = bin_df["spearman_r"].abs()
    bin_df = bin_df.sort_values("abs_spearman_r", ascending=False)
    bin_df.to_csv(out_dir / "binary_target_correlation_ranking.csv", index=False)

    logger.info("Correlation analysis complete → %s", out_dir)
    return results


def _pairwise_significance(df: pd.DataFrame, cols: list, method: str) -> list:
    rows = []
    for c1, c2 in combinations(cols, 2):
        sub = df[[c1, c2]].dropna()
        if len(sub) < 10:
            continue
        if method == "pearson":
            r, p = stats.pearsonr(sub[c1], sub[c2])
        else:
            r, p = stats.spearmanr(sub[c1], sub[c2])
        rows.append({
            "feature_1": c1, "feature_2": c2,
            "r": round(float(r), 4),
            "p_value": round(float(p), 6),
            "n": int(len(sub)),
            "significant_0.05": bool(p < 0.05),
            "significant_0.01": bool(p < 0.01),
            "abs_r": round(abs(float(r)), 4),
        })
    return sorted(rows, key=lambda x: x["abs_r"], reverse=True)


def _plot_heatmap(corr_mat: pd.DataFrame, method: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 8}, linewidths=0.5)
    ax.set_title(f"{method.capitalize()} Correlation Matrix", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(out_dir / f"correlation_{method}_heatmap.png", dpi=150)
    plt.close()


def _plot_target_correlation(target_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, label in [
        (axes[0], "pearson_r", "Pearson r"),
        (axes[1], "spearman_r", "Spearman r"),
    ]:
        df_sorted = target_df.sort_values(col)
        colors = ["#e63946" if v > 0 else "#457b9d" for v in df_sorted[col]]
        ax.barh(df_sorted["feature"], df_sorted[col], color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(label)
        ax.set_title(f"Feature vs Diagnosis ({label})")
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            ax.text(row[col] + (0.01 if row[col] >= 0 else -0.01),
                    i, f"{row[col]:.3f}", va="center",
                    ha="left" if row[col] >= 0 else "right", fontsize=7)
    plt.suptitle("Feature–Target Correlation Ranking (DIAGNOSIS: 1=Control, 2=MCI, 3=Dementia)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "target_feature_correlation_ranking.png", dpi=150)
    plt.close()

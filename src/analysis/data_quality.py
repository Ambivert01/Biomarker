"""
data_quality.py
================
Dataset quality analysis: missingness, sentinels, duplicates, distributions,
outliers, skewness, cardinality. Operates on the REAL ADNI dataset.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

SENTINEL_THRESHOLD = 0.0
BIO_COLS = [
    "pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F",
    "NfL_Q", "GFAP_Q", "NfL_F", "GFAP_F",
]
MMSE_COL = "MMSE_FINAL_SCORE"


def clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel negatives with NaN (mirrors preprocessing pipeline)."""
    df = df.copy()
    for c in BIO_COLS:
        if c in df.columns:
            df.loc[df[c] < SENTINEL_THRESHOLD, c] = np.nan
    if MMSE_COL in df.columns:
        df.loc[(df[MMSE_COL] < 0) | (df[MMSE_COL] > 30), MMSE_COL] = np.nan
    return df


def run_data_quality(df_raw: pd.DataFrame, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = clean_sentinels(df_raw)

    num_cols = [c for c in BIO_COLS + [MMSE_COL] if c in df.columns]
    cat_cols = ["GROUP", "PHASE"]
    id_cols = ["RID", "PTID", "VISCODE", "EXAMDATE"]

    # ── 1. Basic shape ──────────────────────────────────────────────────────
    report = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "n_id_cols": len(id_cols),
        "duplicate_rows": int(df_raw.duplicated().sum()),
        "duplicate_rid": int(df_raw["RID"].duplicated().sum()),
    }

    # ── 2. Missing values ───────────────────────────────────────────────────
    miss_count = df[num_cols].isnull().sum()
    miss_pct = (df[num_cols].isnull().mean() * 100).round(2)
    miss_df = pd.DataFrame({"missing_count": miss_count, "missing_pct": miss_pct})
    miss_df.to_csv(out_dir / "missing_values.csv")
    report["missing_summary"] = miss_df.to_dict()

    # ── 3. Sentinel counts (raw) ────────────────────────────────────────────
    sentinel_rows = []
    for c in BIO_COLS:
        if c in df_raw.columns:
            n_neg = int((df_raw[c] < 0).sum())
            sentinel_rows.append({"column": c, "sentinel_count": n_neg,
                                   "min_raw": float(df_raw[c].min())})
    sentinel_df = pd.DataFrame(sentinel_rows)
    sentinel_df.to_csv(out_dir / "sentinel_counts.csv", index=False)

    # ── 4. Outliers (IQR) ──────────────────────────────────────────────────
    outlier_rows = []
    for c in num_cols:
        s = df[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        outlier_rows.append({
            "column": c, "n_valid": len(s),
            "q1": round(float(q1), 4), "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
            "lower_fence": round(float(lo), 4), "upper_fence": round(float(hi), 4),
            "n_outliers": n_out, "pct_outliers": round(100 * n_out / len(s), 2),
        })
    outlier_df = pd.DataFrame(outlier_rows)
    outlier_df.to_csv(out_dir / "outlier_iqr.csv", index=False)

    # ── 5. Skewness / kurtosis ──────────────────────────────────────────────
    skew_df = pd.DataFrame({
        "skewness": df[num_cols].skew().round(4),
        "kurtosis": df[num_cols].kurtosis().round(4),
    })
    skew_df.to_csv(out_dir / "skewness_kurtosis.csv")

    # ── 6. Categorical cardinality ──────────────────────────────────────────
    cat_rows = []
    for c in cat_cols:
        if c in df.columns:
            vc = df[c].value_counts()
            cat_rows.append({
                "column": c,
                "n_unique": int(df[c].nunique()),
                "mode": str(vc.index[0]),
                "mode_count": int(vc.iloc[0]),
                "mode_pct": round(100 * vc.iloc[0] / len(df), 2),
            })
    pd.DataFrame(cat_rows).to_csv(out_dir / "categorical_cardinality.csv", index=False)

    # ── PLOTS ───────────────────────────────────────────────────────────────
    _plot_missing_bar(miss_df, out_dir)
    _plot_missing_heatmap(df, num_cols, out_dir)
    _plot_target_distribution(df, out_dir)
    _plot_numerical_distributions(df, num_cols, out_dir)
    _plot_boxplots(df, num_cols, out_dir)
    _plot_categorical_freq(df, cat_cols, out_dir)
    _plot_outlier_zscore(df, num_cols, out_dir)

    logger.info("Data quality analysis complete → %s", out_dir)
    return report


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _plot_missing_bar(miss_df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = list(miss_df.index)
    x = range(len(labels))
    colors = ["#e63946" if p > 10 else "#457b9d" for p in miss_df["missing_pct"]]
    ax.bar(x, miss_df["missing_pct"], color=colors, edgecolor="white")
    ax.set_ylabel("Missing (%)")
    ax.set_title("Missing Value Percentage per Feature (after sentinel cleaning)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(5, color="orange", linestyle="--", linewidth=1, label="5% threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "missing_values_bar.png", dpi=150)
    plt.close()


def _plot_missing_heatmap(df: pd.DataFrame, num_cols: list, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    miss_matrix = df[num_cols].isnull().astype(int)
    sns.heatmap(miss_matrix.T, cmap="YlOrRd", cbar=False, ax=ax,
                yticklabels=num_cols, xticklabels=False)
    ax.set_title("Missing Value Pattern (rows=features, cols=patients)")
    ax.set_xlabel("Patients (889)")
    plt.tight_layout()
    plt.savefig(out_dir / "missing_values_heatmap.png", dpi=150)
    plt.close()


def _plot_target_distribution(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vc = df["GROUP"].value_counts()
    colors = ["#2a9d8f", "#e9c46a", "#e63946"]
    axes[0].bar(vc.index, vc.values, color=colors, edgecolor="white")
    for i, (k, v) in enumerate(vc.items()):
        axes[0].text(i, v + 3, f"{v}\n({100*v/len(df):.1f}%)", ha="center", fontsize=9)
    axes[0].set_title("Target Class Distribution (3-class)")
    axes[0].set_ylabel("Count")

    # Binary subset
    binary_df = df[df["DIAGNOSIS"].isin([1, 3])]
    vc2 = binary_df["GROUP"].value_counts()
    colors2 = ["#2a9d8f", "#e63946"]
    axes[1].bar(vc2.index, vc2.values, color=colors2, edgecolor="white")
    for i, (k, v) in enumerate(vc2.items()):
        axes[1].text(i, v + 2, f"{v}\n({100*v/len(binary_df):.1f}%)", ha="center", fontsize=9)
    axes[1].set_title("Binary Subset Distribution (Control vs Dementia)")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "target_distribution.png", dpi=150)
    plt.close()


def _plot_numerical_distributions(df: pd.DataFrame, num_cols: list, out_dir: Path):
    n = len(num_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
    axes = axes.flatten()
    palette = {"Control": "#2a9d8f", "MCI": "#e9c46a", "Dementia": "#e63946"}
    for i, col in enumerate(num_cols):
        ax = axes[i]
        for grp, color in palette.items():
            sub = df[df["GROUP"] == grp][col].dropna()
            if len(sub) > 0:
                ax.hist(sub, bins=30, alpha=0.55, color=color, label=grp, density=True)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Biomarker Distributions by Diagnostic Group", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "numerical_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_boxplots(df: pd.DataFrame, num_cols: list, out_dir: Path):
    n = len(num_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
    axes = axes.flatten()
    palette = {"Control": "#2a9d8f", "MCI": "#e9c46a", "Dementia": "#e63946"}
    for i, col in enumerate(num_cols):
        ax = axes[i]
        data_by_group = [df[df["GROUP"] == g][col].dropna().values
                         for g in ["Control", "MCI", "Dementia"]]
        bp = ax.boxplot(data_by_group, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], palette.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(["Control", "MCI", "Dementia"], fontsize=8)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Biomarker Boxplots by Diagnostic Group", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "boxplots_by_group.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_categorical_freq(df: pd.DataFrame, cat_cols: list, out_dir: Path):
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(14, 4))
    if len(cat_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cat_cols):
        vc = df[col].value_counts()
        ax.bar(vc.index, vc.values, color="#457b9d", edgecolor="white")
        ax.set_title(f"{col} Distribution")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(vc.values):
            ax.text(i, v + 2, str(v), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "categorical_frequency.png", dpi=150)
    plt.close()


def _plot_outlier_zscore(df: pd.DataFrame, num_cols: list, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    z_data = []
    labels = []
    for col in num_cols:
        s = df[col].dropna()
        z = np.abs((s - s.mean()) / s.std())
        z_data.append(z.values)
        labels.append(col)
    ax.boxplot(z_data, patch_artist=True,
               boxprops=dict(facecolor="#457b9d", alpha=0.6))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(3, color="red", linestyle="--", linewidth=1.2, label="|z|=3 threshold")
    ax.set_ylabel("|Z-score|")
    ax.set_title("Outlier Detection via Z-score (|z| > 3 = potential outlier)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "outlier_zscore.png", dpi=150)
    plt.close()

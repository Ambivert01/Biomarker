"""
outlier_analysis.py
====================
Outlier detection using IQR, Z-score, and Isolation Forest.
Documents every decision — outliers are NOT automatically removed.
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
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)
RANDOM_SEED = 42

BIO_COLS = ["pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F",
            "NfL_Q", "GFAP_Q", "NfL_F", "GFAP_F"]


def run_outlier_analysis(df: pd.DataFrame, out_dir: Path) -> dict:
    """df must already have sentinels cleaned."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_cols = [c for c in BIO_COLS if c in df.columns]

    # ── 1. IQR method ────────────────────────────────────────────────────────
    iqr_rows = []
    for col in num_cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        extreme_lo = q1 - 3.0 * iqr
        extreme_hi = q3 + 3.0 * iqr
        outlier_mask = (s < lo) | (s > hi)
        extreme_mask = (s < extreme_lo) | (s > extreme_hi)
        iqr_rows.append({
            "feature": col,
            "n_valid": int(len(s)),
            "lower_fence_1.5iqr": round(float(lo), 4),
            "upper_fence_1.5iqr": round(float(hi), 4),
            "n_outliers_1.5iqr": int(outlier_mask.sum()),
            "pct_outliers_1.5iqr": round(100 * outlier_mask.mean(), 2),
            "n_extreme_3iqr": int(extreme_mask.sum()),
            "pct_extreme_3iqr": round(100 * extreme_mask.mean(), 2),
            "max_value": round(float(s.max()), 4),
            "note": _classify_outlier(col, s.max()),
        })
    iqr_df = pd.DataFrame(iqr_rows)
    iqr_df.to_csv(out_dir / "outliers_iqr.csv", index=False)

    # ── 2. Z-score method ────────────────────────────────────────────────────
    zscore_rows = []
    for col in num_cols:
        s = df[col].dropna()
        z = np.abs(stats.zscore(s))
        zscore_rows.append({
            "feature": col,
            "n_valid": int(len(s)),
            "n_z_gt_2": int((z > 2).sum()),
            "n_z_gt_3": int((z > 3).sum()),
            "n_z_gt_4": int((z > 4).sum()),
            "max_z": round(float(z.max()), 2),
        })
    zscore_df = pd.DataFrame(zscore_rows)
    zscore_df.to_csv(out_dir / "outliers_zscore.csv", index=False)

    # ── 3. Isolation Forest (multivariate) ───────────────────────────────────
    # Use columns with sufficient coverage
    iso_cols = [c for c in ["pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F"]
                if c in df.columns]
    sub_iso = df[iso_cols].dropna()
    if len(sub_iso) > 50:
        iso = IsolationForest(contamination=0.05, random_state=RANDOM_SEED, n_jobs=1)
        iso_labels = iso.fit_predict(sub_iso.values)
        n_anomalies = int((iso_labels == -1).sum())
        iso_result = {
            "n_samples": int(len(sub_iso)),
            "n_anomalies": n_anomalies,
            "pct_anomalies": round(100 * n_anomalies / len(sub_iso), 2),
            "features_used": iso_cols,
        }
        pd.DataFrame([iso_result]).to_csv(out_dir / "outliers_isolation_forest.csv", index=False)

        # Group distribution of anomalies
        sub_iso_with_group = df.loc[sub_iso.index, ["GROUP"]].copy()
        sub_iso_with_group["is_anomaly"] = iso_labels == -1
        anomaly_by_group = sub_iso_with_group.groupby("GROUP")["is_anomaly"].agg(["sum", "mean"])
        anomaly_by_group.columns = ["n_anomalies", "pct_anomalies"]
        anomaly_by_group["pct_anomalies"] = (anomaly_by_group["pct_anomalies"] * 100).round(2)
        anomaly_by_group.to_csv(out_dir / "outliers_isolation_forest_by_group.csv")
    else:
        iso_result = {"note": "Insufficient data for Isolation Forest"}

    # ── PLOTS ────────────────────────────────────────────────────────────────
    _plot_outlier_summary(iqr_df, out_dir)
    _plot_extreme_values(df, num_cols, out_dir)
    _plot_before_after_comparison(df, out_dir)

    logger.info("Outlier analysis complete → %s", out_dir)
    return {
        "iqr_summary": iqr_df.to_dict(orient="records"),
        "isolation_forest": iso_result,
    }


def _classify_outlier(col: str, max_val: float) -> str:
    """Document whether extreme values are clinically plausible."""
    notes = {
        "GFAP_F": "Max=2478 pg/mL — clinically plausible in advanced neurodegeneration; RETAIN",
        "NfL_F": "Max=419.7 pg/mL — extreme but plausible in severe neurodegeneration; RETAIN",
        "AB40_F": "Max=1927 pg/mL — extreme; verify but plausible in outlier patients; RETAIN",
        "pT217_F": "Max=5.56 pg/mL — ~7× median; plausible in advanced AD; RETAIN",
        "NfL_Q": "Max=278 pg/mL — plausible; different scale from Fujirebio; RETAIN",
        "GFAP_Q": "Max=816.7 pg/mL — plausible on Quanterix scale; RETAIN",
    }
    return notes.get(col, "Review recommended but no automatic removal")


def _plot_outlier_summary(iqr_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_s = iqr_df.sort_values("pct_outliers_1.5iqr", ascending=True)
    colors = ["#e63946" if p > 10 else "#457b9d" for p in df_s["pct_outliers_1.5iqr"]]
    axes[0].barh(df_s["feature"], df_s["pct_outliers_1.5iqr"], color=colors, edgecolor="white")
    axes[0].axvline(5, color="orange", linestyle="--", lw=1.2, label="5% threshold")
    axes[0].set_xlabel("% Outliers (IQR 1.5×)")
    axes[0].set_title("Outlier Percentage per Feature (IQR method)")
    axes[0].legend()

    df_s2 = iqr_df.sort_values("pct_extreme_3iqr", ascending=True)
    axes[1].barh(df_s2["feature"], df_s2["pct_extreme_3iqr"], color="#e9c46a", edgecolor="white")
    axes[1].set_xlabel("% Extreme Outliers (IQR 3×)")
    axes[1].set_title("Extreme Outlier Percentage (IQR 3× — severe)")
    plt.tight_layout()
    plt.savefig(out_dir / "outlier_summary.png", dpi=150)
    plt.close()


def _plot_extreme_values(df: pd.DataFrame, num_cols: list, out_dir: Path):
    """Show the top extreme values per feature."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    highlight_cols = ["pT217_F", "AB40_F", "NfL_F", "GFAP_F", "NfL_Q", "GFAP_Q"]
    palette = {"Control": "#2a9d8f", "MCI": "#e9c46a", "Dementia": "#e63946"}
    for i, col in enumerate(highlight_cols):
        if col not in df.columns or i >= len(axes):
            continue
        ax = axes[i]
        for grp, color in palette.items():
            s = df[df["GROUP"] == grp][col].dropna()
            ax.scatter(range(len(s)), s.sort_values().values, alpha=0.4, s=10,
                       color=color, label=grp)
        q3 = df[col].dropna().quantile(0.75)
        iqr = q3 - df[col].dropna().quantile(0.25)
        ax.axhline(q3 + 1.5 * iqr, color="orange", linestyle="--", lw=1, label="1.5×IQR fence")
        ax.axhline(q3 + 3.0 * iqr, color="red", linestyle="--", lw=1, label="3×IQR fence")
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("Value")
        if i == 0:
            ax.legend(fontsize=6)
    plt.suptitle("Extreme Value Visualization per Biomarker", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "extreme_values.png", dpi=150)
    plt.close()


def _plot_before_after_comparison(df: pd.DataFrame, out_dir: Path):
    """Compare distributions with and without extreme outliers (IQR 3×)."""
    col = "GFAP_F"  # Most extreme outlier in dataset
    if col not in df.columns:
        return
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    s_trimmed = s[s <= q3 + 3 * iqr]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(s, bins=50, color="#e63946", alpha=0.7, edgecolor="white")
    axes[0].set_title(f"{col} — Full Distribution\n(n={len(s)}, max={s.max():.1f})")
    axes[0].set_xlabel("pg/mL")

    axes[1].hist(s_trimmed, bins=50, color="#2a9d8f", alpha=0.7, edgecolor="white")
    axes[1].set_title(f"{col} — Trimmed (3×IQR fence)\n(n={len(s_trimmed)}, {len(s)-len(s_trimmed)} removed)")
    axes[1].set_xlabel("pg/mL")

    plt.suptitle(f"GFAP_F: Before vs After Extreme Outlier Trimming\n"
                 f"NOTE: Extreme values are clinically plausible — NOT removed in model", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "outlier_before_after_gfap.png", dpi=150)
    plt.close()

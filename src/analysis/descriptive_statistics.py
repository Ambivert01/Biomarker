"""
descriptive_statistics.py
==========================
Full descriptive statistics for all numerical and categorical features
in the ADNI dataset. Exports CSV tables to results/statistics/.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)

BIO_COLS = [
    "pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F",
    "NfL_Q", "GFAP_Q", "NfL_F", "GFAP_F",
]
MMSE_COL = "MMSE_FINAL_SCORE"
CAT_COLS = ["GROUP", "PHASE"]
GROUPS = ["Control", "MCI", "Dementia"]


def run_descriptive_statistics(df: pd.DataFrame, out_dir: Path) -> dict:
    """
    Compute and export full descriptive statistics.
    df must already have sentinels cleaned (negatives → NaN).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_cols = [c for c in BIO_COLS + [MMSE_COL] if c in df.columns]

    # ── 1. Overall numerical stats ──────────────────────────────────────────
    rows = []
    for col in num_cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append({
            "feature": col,
            "count": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "variance": round(float(s.var()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "range": round(float(s.max() - s.min()), 4),
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(q3 - q1), 4),
            "skewness": round(float(stats.skew(s)), 4),
            "kurtosis": round(float(stats.kurtosis(s)), 4),
            "missing_count": int(df[col].isnull().sum()),
            "missing_pct": round(100 * df[col].isnull().mean(), 2),
        })
    overall_df = pd.DataFrame(rows)
    overall_df.to_csv(out_dir / "descriptive_stats_overall.csv", index=False)

    # ── 2. Per-group numerical stats ────────────────────────────────────────
    group_rows = []
    for grp in GROUPS:
        sub = df[df["GROUP"] == grp]
        for col in num_cols:
            s = sub[col].dropna()
            if len(s) < 2:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            group_rows.append({
                "group": grp,
                "feature": col,
                "n": int(len(s)),
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "iqr": round(float(q3 - q1), 4),
                "skewness": round(float(stats.skew(s)), 4),
                "kurtosis": round(float(stats.kurtosis(s)), 4),
                "p10": round(float(s.quantile(0.10)), 4),
                "p90": round(float(s.quantile(0.90)), 4),
            })
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(out_dir / "descriptive_stats_by_group.csv", index=False)

    # ── 3. Categorical stats ────────────────────────────────────────────────
    cat_rows = []
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        for val, cnt in vc.items():
            cat_rows.append({
                "column": col,
                "value": val,
                "count": int(cnt),
                "percentage": round(100 * cnt / len(df), 2),
            })
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(out_dir / "categorical_stats.csv", index=False)

    # ── 4. MMSE by group ────────────────────────────────────────────────────
    mmse_rows = []
    for grp in GROUPS:
        s = df[df["GROUP"] == grp][MMSE_COL].dropna()
        if len(s) < 2:
            continue
        mmse_rows.append({
            "group": grp,
            "n_with_mmse": int(len(s)),
            "n_missing_mmse": int((df["GROUP"] == grp).sum() - len(s)),
            "pct_missing": round(100 * ((df["GROUP"] == grp).sum() - len(s)) / (df["GROUP"] == grp).sum(), 1),
            "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 1),
            "max": round(float(s.max()), 1),
        })
    pd.DataFrame(mmse_rows).to_csv(out_dir / "mmse_by_group.csv", index=False)

    # ── 5. Phase × Group cross-tab ──────────────────────────────────────────
    crosstab = pd.crosstab(df["PHASE"], df["GROUP"])
    crosstab.to_csv(out_dir / "phase_group_crosstab.csv")
    crosstab_pct = pd.crosstab(df["PHASE"], df["GROUP"], normalize="index").round(4) * 100
    crosstab_pct.to_csv(out_dir / "phase_group_crosstab_pct.csv")

    logger.info("Descriptive statistics complete → %s", out_dir)
    return {
        "overall_stats": overall_df.to_dict(orient="records"),
        "group_stats": group_df.to_dict(orient="records"),
    }

"""
statistical_tests.py
=====================
Group comparison tests: Kruskal-Wallis, Mann-Whitney U, Welch ANOVA,
chi-square, and pairwise post-hoc tests. All tests chosen based on
data characteristics (non-normal distributions, ordinal groups).
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
GROUPS = ["Control", "MCI", "Dementia"]


def run_statistical_tests(df: pd.DataFrame, out_dir: Path) -> dict:
    """
    df must already have sentinels cleaned.
    Runs appropriate statistical tests for group comparisons.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_cols = [c for c in BIO_COLS + [MMSE_COL] if c in df.columns]

    # ── 1. Normality check (Shapiro-Wilk on sample ≤ 5000) ─────────────────
    norm_rows = []
    for col in num_cols:
        for grp in GROUPS:
            s = df[df["GROUP"] == grp][col].dropna()
            if len(s) < 8:
                continue
            sample = s.sample(min(len(s), 500), random_state=42)
            stat, p = stats.shapiro(sample)
            norm_rows.append({
                "feature": col, "group": grp, "n": int(len(s)),
                "shapiro_stat": round(float(stat), 4),
                "shapiro_p": round(float(p), 6),
                "normal_at_0.05": bool(p >= 0.05),
            })
    norm_df = pd.DataFrame(norm_rows)
    norm_df.to_csv(out_dir / "normality_shapiro.csv", index=False)

    # ── 2. Kruskal-Wallis (3-group, non-parametric) ─────────────────────────
    # Appropriate because: biomarkers are right-skewed, non-normal (confirmed above)
    kw_rows = []
    for col in num_cols:
        groups_data = [df[df["GROUP"] == g][col].dropna().values for g in GROUPS]
        groups_data = [g for g in groups_data if len(g) >= 5]
        if len(groups_data) < 2:
            continue
        stat, p = stats.kruskal(*groups_data)
        # Effect size: eta-squared approximation
        n_total = sum(len(g) for g in groups_data)
        eta2 = (stat - len(groups_data) + 1) / (n_total - len(groups_data))
        kw_rows.append({
            "feature": col,
            "kruskal_H": round(float(stat), 4),
            "p_value": round(float(p), 8),
            "eta_squared": round(float(max(0, eta2)), 4),
            "significant_0.05": bool(p < 0.05),
            "significant_0.001": bool(p < 0.001),
            "n_total": int(n_total),
        })
    kw_df = pd.DataFrame(kw_rows).sort_values("p_value")
    kw_df.to_csv(out_dir / "kruskal_wallis_3group.csv", index=False)
    _plot_kruskal_summary(kw_df, out_dir)

    # ── 3. Pairwise Mann-Whitney U (with Bonferroni correction) ─────────────
    mw_rows = []
    pairs = list(combinations(GROUPS, 2))
    n_tests = len(num_cols) * len(pairs)
    for col in num_cols:
        for g1, g2 in pairs:
            s1 = df[df["GROUP"] == g1][col].dropna().values
            s2 = df[df["GROUP"] == g2][col].dropna().values
            if len(s1) < 5 or len(s2) < 5:
                continue
            stat, p = stats.mannwhitneyu(s1, s2, alternative="two-sided")
            # Effect size r = Z / sqrt(N)
            z = stats.norm.ppf(1 - p / 2)
            r = z / np.sqrt(len(s1) + len(s2))
            mw_rows.append({
                "feature": col, "group_1": g1, "group_2": g2,
                "n_1": int(len(s1)), "n_2": int(len(s2)),
                "U_stat": round(float(stat), 2),
                "p_value": round(float(p), 8),
                "p_bonferroni": round(float(min(p * n_tests, 1.0)), 8),
                "effect_r": round(float(r), 4),
                "significant_0.05": bool(p < 0.05),
                "significant_bonferroni": bool(p * n_tests < 0.05),
            })
    mw_df = pd.DataFrame(mw_rows).sort_values("p_value")
    mw_df.to_csv(out_dir / "mannwhitney_pairwise.csv", index=False)

    # ── 4. Binary group comparison (Control vs Dementia) ────────────────────
    bin_rows = []
    df_bin = df[df["DIAGNOSIS"].isin([1, 3])]
    for col in num_cols:
        s_ctrl = df_bin[df_bin["GROUP"] == "Control"][col].dropna().values
        s_dem = df_bin[df_bin["GROUP"] == "Dementia"][col].dropna().values
        if len(s_ctrl) < 5 or len(s_dem) < 5:
            continue
        stat_mw, p_mw = stats.mannwhitneyu(s_ctrl, s_dem, alternative="two-sided")
        stat_t, p_t = stats.ttest_ind(s_ctrl, s_dem, equal_var=False)  # Welch's t-test
        # Cohen's d
        pooled_std = np.sqrt((s_ctrl.std() ** 2 + s_dem.std() ** 2) / 2)
        cohens_d = (s_dem.mean() - s_ctrl.mean()) / pooled_std if pooled_std > 0 else 0
        # Fold change
        fold_change = s_dem.mean() / s_ctrl.mean() if s_ctrl.mean() != 0 else np.nan
        bin_rows.append({
            "feature": col,
            "control_mean": round(float(s_ctrl.mean()), 4),
            "dementia_mean": round(float(s_dem.mean()), 4),
            "fold_change_dem_ctrl": round(float(fold_change), 4),
            "welch_t": round(float(stat_t), 4),
            "welch_p": round(float(p_t), 8),
            "mannwhitney_U": round(float(stat_mw), 2),
            "mannwhitney_p": round(float(p_mw), 8),
            "cohens_d": round(float(cohens_d), 4),
            "n_control": int(len(s_ctrl)),
            "n_dementia": int(len(s_dem)),
        })
    bin_df = pd.DataFrame(bin_rows).sort_values("mannwhitney_p")
    bin_df.to_csv(out_dir / "binary_group_comparison.csv", index=False)
    _plot_binary_comparison(bin_df, out_dir)

    # ── 5. Chi-square: PHASE × GROUP ────────────────────────────────────────
    ct = pd.crosstab(df["PHASE"], df["GROUP"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
    # Cramér's V
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    chi2_result = {
        "test": "chi2_phase_vs_group",
        "chi2_stat": round(float(chi2), 4),
        "p_value": round(float(p_chi2), 8),
        "dof": int(dof),
        "cramers_v": round(float(cramers_v), 4),
        "n": int(n),
        "interpretation": "PHASE and GROUP are NOT independent (p<0.001) — major confound" if p_chi2 < 0.001 else "No significant association",
    }
    pd.DataFrame([chi2_result]).to_csv(out_dir / "chi2_phase_group.csv", index=False)

    # ── 6. MMSE group comparison ─────────────────────────────────────────────
    _plot_mmse_by_group(df, out_dir)

    logger.info("Statistical tests complete → %s", out_dir)
    return {
        "kruskal_wallis": kw_df.to_dict(orient="records"),
        "binary_comparison": bin_df.to_dict(orient="records"),
        "chi2_phase_group": chi2_result,
    }


def _plot_kruskal_summary(kw_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_sorted = kw_df.sort_values("kruskal_H", ascending=True)
    colors = ["#e63946" if p < 0.001 else "#457b9d" for p in df_sorted["p_value"]]
    axes[0].barh(df_sorted["feature"], df_sorted["kruskal_H"], color=colors, edgecolor="white")
    axes[0].set_xlabel("Kruskal-Wallis H statistic")
    axes[0].set_title("Kruskal-Wallis H (3-group: Control/MCI/Dementia)\nRed = p<0.001")

    df_sorted2 = kw_df.sort_values("eta_squared", ascending=True)
    axes[1].barh(df_sorted2["feature"], df_sorted2["eta_squared"],
                 color="#2a9d8f", edgecolor="white")
    axes[1].set_xlabel("Eta-squared (effect size)")
    axes[1].set_title("Effect Size (η²) per Biomarker")
    plt.tight_layout()
    plt.savefig(out_dir / "kruskal_wallis_summary.png", dpi=150)
    plt.close()


def _plot_binary_comparison(bin_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_sorted = bin_df.sort_values("cohens_d", ascending=True)
    colors = ["#e63946" if d > 0 else "#457b9d" for d in df_sorted["cohens_d"]]
    axes[0].barh(df_sorted["feature"], df_sorted["cohens_d"], color=colors, edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Cohen's d (Dementia − Control)")
    axes[0].set_title("Effect Size: Control vs Dementia\n(positive = higher in Dementia)")

    df_sorted2 = bin_df.sort_values("fold_change_dem_ctrl", ascending=True)
    axes[1].barh(df_sorted2["feature"], df_sorted2["fold_change_dem_ctrl"],
                 color="#e9c46a", edgecolor="white")
    axes[1].axvline(1, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Fold Change (Dementia mean / Control mean)")
    axes[1].set_title("Fold Change: Dementia vs Control")
    plt.tight_layout()
    plt.savefig(out_dir / "binary_group_comparison.png", dpi=150)
    plt.close()


def _plot_mmse_by_group(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    palette = {"Control": "#2a9d8f", "MCI": "#e9c46a", "Dementia": "#e63946"}
    # Violin
    data_by_group = [df[df["GROUP"] == g]["MMSE_FINAL_SCORE"].dropna().values
                     for g in ["Control", "MCI", "Dementia"]]
    parts = axes[0].violinplot(data_by_group, positions=[0, 1, 2], showmedians=True)
    for i, (pc, color) in enumerate(zip(parts["bodies"], palette.values())):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["Control", "MCI", "Dementia"])
    axes[0].set_ylabel("MMSE Score")
    axes[0].set_title("MMSE Distribution by Group")

    # Missing MMSE bar
    miss_pct = [100 * df[df["GROUP"] == g]["MMSE_FINAL_SCORE"].isnull().mean()
                for g in ["Control", "MCI", "Dementia"]]
    axes[1].bar(["Control", "MCI", "Dementia"], miss_pct,
                color=list(palette.values()), edgecolor="white")
    for i, v in enumerate(miss_pct):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    axes[1].set_ylabel("MMSE Missing (%)")
    axes[1].set_title("MMSE Missingness by Group\n(non-random — clinical workflow artifact)")
    plt.tight_layout()
    plt.savefig(out_dir / "mmse_by_group.png", dpi=150)
    plt.close()

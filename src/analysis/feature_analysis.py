"""
feature_analysis.py
====================
Feature importance, permutation importance, univariate analysis,
bivariate analysis, and multivariate analysis (PCA, pair plots).
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
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib

logger = logging.getLogger(__name__)
RANDOM_SEED = 42

_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

BIO_COLS = ["pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F",
            "NfL_harmonized", "GFAP_harmonized"]
GROUPS = ["Control", "MCI", "Dementia"]


def _harmonize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    nfl_ratio, gfap_ratio = 1.465, 0.376
    nfl_q_as_f = df["NfL_Q"] * nfl_ratio
    df["NfL_harmonized"] = np.where(
        df["NfL_F"].notna() & df["NfL_Q"].notna(), (df["NfL_F"] + nfl_q_as_f) / 2,
        np.where(df["NfL_F"].notna(), df["NfL_F"],
                 np.where(df["NfL_Q"].notna(), nfl_q_as_f, np.nan)))
    gfap_q_as_f = df["GFAP_Q"] * gfap_ratio
    df["GFAP_harmonized"] = np.where(
        df["GFAP_F"].notna() & df["GFAP_Q"].notna(), (df["GFAP_F"] + gfap_q_as_f) / 2,
        np.where(df["GFAP_F"].notna(), df["GFAP_F"],
                 np.where(df["GFAP_Q"].notna(), gfap_q_as_f, np.nan)))
    return df


def run_feature_analysis(df_clean: pd.DataFrame, out_dir: Path, model_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)

    df = _harmonize(df_clean)

    # 1. SHAP importance from existing report
    shap_path = Path(__file__).resolve().parent.parent.parent / "reports" / "shap_global_importance.csv"
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        _plot_shap_importance(shap_df, out_dir)

    # 2. Built-in RF feature importance
    binary_path = model_dir / "final_model_binary_uncalibrated.joblib"
    if binary_path.exists():
        _extract_rf_importance(binary_path, out_dir)

    # 3. Univariate analysis
    _univariate_analysis(df, out_dir)

    # 4. Bivariate: numerical vs numerical
    _bivariate_num_num(df, out_dir)

    # 5. Bivariate: numerical vs categorical
    _bivariate_num_cat(df, out_dir)

    # 6. PCA
    _pca_analysis(df, out_dir)

    # 7. Pair plot
    _pair_plot(df, out_dir)

    logger.info("Feature analysis complete → %s", out_dir)
    return {}


def _plot_shap_importance(shap_df: pd.DataFrame, out_dir: Path):
    df_s = shap_df.sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#e63946" if "pT217" in f else ("#457b9d" if "NfL" in f else "#2a9d8f")
              for f in df_s["feature"]]
    ax.barh(df_s["feature"], df_s["mean_abs_shap"], color=colors, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global SHAP Feature Importance — Binary Model (Random Forest)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_feature_importance.png", dpi=150)
    plt.close()


def _extract_rf_importance(model_path: Path, out_dir: Path):
    try:
        pipe = joblib.load(model_path)
        clf = pipe.named_steps["clf"]
        feature_names = [
            "pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F",
            "NfL_harmonized", "GFAP_harmonized", "GFAP_NfL_ratio",
            "AB40_minus_AB42", "pT217_GFAP_product", "log_pT217_F",
            "log_AB40_F", "log_NfL_harmonized", "log_GFAP_harmonized",
            "rank_pT217_F", "rank_AB42_F", "rank_AB40_F",
            "rank_NfL_harmonized", "rank_GFAP_harmonized",
        ]
        importances = clf.feature_importances_
        n = min(len(feature_names), len(importances))
        imp_df = pd.DataFrame({
            "feature": feature_names[:n],
            "importance": importances[:n],
        }).sort_values("importance", ascending=False)
        imp_df.to_csv(out_dir / "rf_feature_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(9, 7))
        df_s = imp_df.sort_values("importance", ascending=True)
        ax.barh(df_s["feature"], df_s["importance"], color="#457b9d", edgecolor="white")
        ax.set_xlabel("Feature Importance (Gini impurity)")
        ax.set_title("Random Forest Built-in Feature Importance — Binary Model")
        plt.tight_layout()
        plt.savefig(out_dir / "rf_feature_importance.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not extract RF importance: %s", e)


def _univariate_analysis(df: pd.DataFrame, out_dir: Path):
    core_cols = [c for c in BIO_COLS if c in df.columns]
    palette = {"Control": "#2a9d8f", "MCI": "#e9c46a", "Dementia": "#e63946"}

    for col in core_cols:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for grp, color in palette.items():
            sub = df[df["GROUP"] == grp][col].dropna()
            axes[0].hist(sub, bins=25, alpha=0.55, color=color, label=grp, density=True)
        axes[0].set_title(f"{col} — Distribution by Group")
        axes[0].set_xlabel(col)
        axes[0].legend(fontsize=8)

        data = [df[df["GROUP"] == g][col].dropna().values for g in GROUPS]
        parts = axes[1].violinplot(data, positions=[0, 1, 2], showmedians=True)
        for pc, color in zip(parts["bodies"], palette.values()):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        axes[1].set_xticks([0, 1, 2])
        axes[1].set_xticklabels(GROUPS)
        axes[1].set_title(f"{col} — Violin by Group")

        log_col = np.log1p(df[col].clip(lower=0))
        for grp, color in palette.items():
            sub = log_col[df["GROUP"] == grp].dropna()
            axes[2].hist(sub, bins=25, alpha=0.55, color=color, label=grp, density=True)
        axes[2].set_title(f"log1p({col}) — Distribution")
        axes[2].set_xlabel(f"log1p({col})")

        plt.suptitle(f"Univariate Analysis: {col}", fontsize=11)
        plt.tight_layout()
        safe = col.replace("/", "_")
        plt.savefig(out_dir / f"univariate_{safe}.png", dpi=150)
        plt.close()


def _bivariate_num_num(df: pd.DataFrame, out_dir: Path):
    pairs = [
        ("pT217_F", "NfL_harmonized"),
        ("pT217_F", "GFAP_harmonized"),
        ("pT217_F", "AB42_AB40_F"),
        ("NfL_harmonized", "GFAP_harmonized"),
        ("AB42_F", "AB40_F"),
    ]
    palette = {"Control": "#2a9d8f", "MCI": "#e9c46a", "Dementia": "#e63946"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, (cx, cy) in enumerate(pairs):
        if cx not in df.columns or cy not in df.columns:
            continue
        ax = axes[i]
        for grp, color in palette.items():
            sub = df[df["GROUP"] == grp][[cx, cy]].dropna()
            ax.scatter(sub[cx], sub[cy], alpha=0.4, s=15, color=color, label=grp)
        ax.set_xlabel(cx, fontsize=8)
        ax.set_ylabel(cy, fontsize=8)
        ax.set_title(f"{cx} vs {cy}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
    axes[-1].set_visible(False)
    plt.suptitle("Bivariate Scatter Plots (colored by diagnostic group)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "bivariate_scatter_plots.png", dpi=150)
    plt.close()


def _bivariate_num_cat(df: pd.DataFrame, out_dir: Path):
    core_cols = [c for c in BIO_COLS if c in df.columns]
    palette = {"Control": "#2a9d8f", "MCI": "#e9c46a", "Dementia": "#e63946"}
    n = len(core_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()
    for i, col in enumerate(core_cols):
        ax = axes[i]
        data = [df[df["GROUP"] == g][col].dropna().values for g in GROUPS]
        parts = ax.violinplot(data, positions=[0, 1, 2], showmedians=True)
        for pc, color in zip(parts["bodies"], palette.values()):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(GROUPS, fontsize=8)
        ax.set_title(col, fontsize=9)
        for j, grp in enumerate(GROUPS):
            m = df[df["GROUP"] == grp][col].median()
            if not np.isnan(m):
                ax.text(j, m, f"{m:.2f}", ha="center", va="bottom", fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Biomarker Distributions by Diagnostic Group (Violin)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "bivariate_violin_by_group.png", dpi=150)
    plt.close()


def _pca_analysis(df: pd.DataFrame, out_dir: Path):
    core_cols = [c for c in BIO_COLS if c in df.columns]
    sub = df[core_cols + ["GROUP"]].dropna()
    X = sub[core_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    n_comp = min(len(core_cols), 5)
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    ev = pca.explained_variance_ratio_
    pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(ev))],
        "explained_variance_ratio": ev.round(4),
        "cumulative": np.cumsum(ev).round(4),
    }).to_csv(out_dir / "pca_explained_variance.csv", index=False)

    pd.DataFrame(
        pca.components_.T,
        index=core_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    ).to_csv(out_dir / "pca_loadings.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(range(1, len(ev) + 1), ev * 100, color="#457b9d", edgecolor="white")
    axes[0].plot(range(1, len(ev) + 1), np.cumsum(ev) * 100, "ro-", lw=2)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("PCA Scree Plot")

    palette = {"Control": "#2a9d8f", "MCI": "#e9c46a", "Dementia": "#e63946"}
    for grp, color in palette.items():
        mask = sub["GROUP"].values == grp
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.5, s=20, color=color, label=grp)
    axes[1].set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    axes[1].set_title("PCA: PC1 vs PC2 by Group")
    axes[1].legend(fontsize=8)

    loadings = pd.DataFrame(pca.components_.T, index=core_cols,
                             columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    sns.heatmap(loadings.iloc[:, :3], annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=axes[2])
    axes[2].set_title("PCA Loadings (PC1–PC3)")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_analysis.png", dpi=150)
    plt.close()


def _pair_plot(df: pd.DataFrame, out_dir: Path):
    cols = [c for c in ["pT217_F", "AB42_AB40_F", "NfL_harmonized", "GFAP_harmonized"]
            if c in df.columns]
    sub = df[cols + ["GROUP"]].dropna()
    if len(sub) > 400:
        sub = sub.sample(400, random_state=RANDOM_SEED)
    palette = {"Control": "#2a9d8f", "MCI": "#e9c46a", "Dementia": "#e63946"}
    g = sns.pairplot(sub, hue="GROUP", palette=palette,
                     plot_kws={"alpha": 0.4, "s": 15},
                     diag_kind="kde", corner=True)
    g.fig.suptitle("Pair Plot — Core Biomarkers by Diagnostic Group", y=1.01, fontsize=11)
    plt.savefig(out_dir / "pairplot_core_biomarkers.png", dpi=150, bbox_inches="tight")
    plt.close()

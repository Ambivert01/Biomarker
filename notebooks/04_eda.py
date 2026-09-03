import sys, warnings
sys.path.append('../src')
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 110
FIGDIR = "../reports/figures"
GROUP_ORDER = ["Control", "MCI", "Dementia"]
PALETTE = {"Control": "#2E7D32", "MCI": "#F9A825", "Dementia": "#C62828"}

df = pd.read_parquet("../data/EDA_clean_full.parquet")
df["GROUP"] = pd.Categorical(df["GROUP"], categories=GROUP_ORDER, ordered=True)

# 1. Target distribution -----------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
counts = df["GROUP"].value_counts().reindex(GROUP_ORDER)
bars = ax.bar(counts.index, counts.values, color=[PALETTE[g] for g in GROUP_ORDER])
for b, c in zip(bars, counts.values):
    ax.text(b.get_x()+b.get_width()/2, c+3, f"{c}\n({100*c/len(df):.1f}%)", ha='center', fontsize=10)
ax.set_title("Target Distribution — Diagnosis (n=889)")
ax.set_ylabel("Patients")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/01_target_distribution.png"); plt.close()

# 2. Biomarker distributions by group (violin + box) -------------------------
biomarkers = ["pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F", "NfL_harmonized", "GFAP_harmonized"]
fig, axes = plt.subplots(3, 3, figsize=(15, 13))
axes = axes.flatten()
for i, col in enumerate(biomarkers):
    sns.violinplot(data=df, x="GROUP", y=col, order=GROUP_ORDER, palette=PALETTE, ax=axes[i], cut=0, inner="quartile")
    axes[i].set_title(col)
    axes[i].set_xlabel("")
sns.violinplot(data=df, x="GROUP", y="MMSE_FINAL_SCORE", order=GROUP_ORDER, palette=PALETTE, ax=axes[7], cut=0, inner="quartile")
axes[7].set_title("MMSE_FINAL_SCORE")
axes[8].axis('off')
plt.suptitle("Biomarker & MMSE Distributions by Diagnostic Group", y=1.01, fontsize=14)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/02_biomarker_violin_by_group.png", bbox_inches='tight'); plt.close()

# 3. Log-scale versions for the heavily skewed ones --------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for i, col in enumerate(["log_pT217_F", "log_AB40_F", "log_NfL_harmonized", "log_GFAP_harmonized"]):
    sns.boxplot(data=df, x="GROUP", y=col, order=GROUP_ORDER, palette=PALETTE, ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/03_log_biomarker_boxplots.png"); plt.close()

# 4. Correlation heatmap ------------------------------------------------------
corr_cols = biomarkers + ["MMSE_FINAL_SCORE", "GFAP_NfL_ratio", "AB40_minus_AB42"]
corr = df[corr_cols].corr(method='spearman')
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax,
            cbar_kws={'label': 'Spearman r'})
ax.set_title("Feature Correlation Heatmap (Spearman)")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/04_correlation_heatmap.png"); plt.close()

# 5. Missingness heatmap (raw, pre-cleaning columns) --------------------------
raw_cols = ["pT217_F","AB42_F","AB40_F","AB42_AB40_F","pT217_AB42_F","NfL_Q","GFAP_Q","NfL_F","GFAP_F","MMSE_FINAL_SCORE"]
miss_mask = df[raw_cols].isna() | (df[raw_cols] < 0)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(miss_mask.astype(int).sample(200, random_state=42).sort_values(by=raw_cols[0]) if False else miss_mask.astype(int),
            cbar_kws={'label': 'Missing / BLoQ sentinel'}, cmap=["#EEEEEE", "#C62828"], ax=ax, yticklabels=False)
ax.set_title("Missingness / BLoQ-Sentinel Map (raw columns, 889 patients)")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/05_missingness_heatmap.png"); plt.close()

# 6. Missingness bar chart ----------------------------------------------------
miss_pct = (miss_mask.mean()*100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(miss_pct.index, miss_pct.values, color="#C62828")
for i, v in enumerate(miss_pct.values):
    ax.text(v+0.5, i, f"{v:.1f}%", va='center')
ax.set_xlabel("% missing or BLoQ sentinel")
ax.set_title("Missingness / Sentinel Rate by Raw Column")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/06_missingness_barchart.png"); plt.close()

# 7. PHASE x GROUP confound plot ----------------------------------------------
ct = pd.crosstab(df["PHASE"], df["GROUP"], normalize='index')[GROUP_ORDER]
fig, ax = plt.subplots(figsize=(8, 5))
ct.plot(kind='bar', stacked=True, color=[PALETTE[g] for g in GROUP_ORDER], ax=ax)
ax.set_ylabel("Proportion of patients")
ax.set_title("Diagnosis Mix by ADNI Phase — Major Cohort Confound\n(chi2 p=1.3e-47, see DATA_AUDIT_REPORT.md \u00a73.3)")
ax.legend(title="Diagnosis", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout(); plt.savefig(f"{FIGDIR}/07_phase_group_confound.png"); plt.close()

# 8. Assay platform agreement scatter -----------------------------------------
both = df[(df["NfL_F"].notna()) & (df["NfL_Q"].notna()) & (df["NfL_F"]>0) & (df["NfL_Q"]>0)]
both_g = df[(df["GFAP_F"].notna()) & (df["GFAP_Q"].notna()) & (df["GFAP_F"]>0) & (df["GFAP_Q"]>0)]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, xcol, ycol, title in [
    (axes[0], both, "NfL_Q", "NfL_F", "NfL: Fujirebio vs Quanterix"),
    (axes[1], both_g, "GFAP_Q", "GFAP_F", "GFAP: Fujirebio vs Quanterix"),
]:
    sns.scatterplot(data=data, x=xcol, y=ycol, hue="GROUP", palette=PALETTE, hue_order=GROUP_ORDER, ax=ax, alpha=0.75)
    lims = [0, max(data[xcol].max(), data[ycol].max())*1.05]
    r = data[xcol].corr(data[ycol])
    ax.set_title(f"{title}\nPearson r={r:.2f}, n={len(data)}")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/08_assay_platform_agreement.png"); plt.close()

# 9. Pairplot of core biomarkers ----------------------------------------------
pp_cols = ["pT217_F", "AB42_AB40_F", "NfL_harmonized", "GFAP_harmonized", "GROUP"]
g = sns.pairplot(df[pp_cols].dropna(), hue="GROUP", hue_order=GROUP_ORDER, palette=PALETTE,
                  plot_kws={'alpha': 0.55, 's': 20}, diag_kind='kde', corner=True)
g.fig.suptitle("Pairwise Relationships — Core Biomarkers", y=1.02)
g.savefig(f"{FIGDIR}/09_pairplot_core_biomarkers.png", bbox_inches='tight')
plt.close('all')

# 10. Outlier boxplots (z-score view for all core biomarkers) -----------------
fig, ax = plt.subplots(figsize=(10, 5))
zdf = df[biomarkers].apply(lambda s: (s - s.mean())/s.std())
zdf_melt = zdf.melt(var_name="feature", value_name="zscore")
sns.boxplot(data=zdf_melt, x="feature", y="zscore", ax=ax)
ax.axhline(3, color='red', ls='--', lw=1); ax.axhline(-3, color='red', ls='--', lw=1)
ax.set_title("Outlier View (z-scores; dashed lines = \u00b13\u03c3)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout(); plt.savefig(f"{FIGDIR}/10_outlier_zscores.png"); plt.close()

# 11. Mutual information & ANOVA F-test with target ---------------------------
mi_cols = biomarkers + ["MMSE_FINAL_SCORE", "GFAP_NfL_ratio", "AB40_minus_AB42"]
tmp = df[mi_cols + ["DIAGNOSIS"]].dropna()
mi = mutual_info_classif(tmp[mi_cols], tmp["DIAGNOSIS"], random_state=42)
fvals, pvals = f_classif(tmp[mi_cols], tmp["DIAGNOSIS"])
imp_df = pd.DataFrame({"feature": mi_cols, "mutual_info": mi, "anova_F": fvals, "anova_p": pvals}).sort_values("mutual_info", ascending=False)
imp_df.to_csv("../reports/feature_importance_univariate.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp_df["feature"], imp_df["mutual_info"], color="#1565C0")
ax.invert_yaxis()
ax.set_xlabel("Mutual Information with DIAGNOSIS")
ax.set_title("Univariate Feature Importance (Mutual Information, complete-case n={})".format(len(tmp)))
plt.tight_layout(); plt.savefig(f"{FIGDIR}/11_mutual_information.png"); plt.close()

print(imp_df.to_string(index=False))
print("\nAll EDA figures written to", FIGDIR)

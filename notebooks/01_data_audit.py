"""
DATA AUDIT — verifies documentation claims against the actual Excel data,
and performs a full integrity/quality sweep across every sheet.
Run: python3 01_data_audit.py > ../reports/data_audit_raw.txt
"""
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 220)

D = '../data/'
diag1 = pd.read_parquet(D+'Diagnosis_1.parquet')
diag2 = pd.read_parquet(D+'Diagnosis_2.parquet')
diag3 = pd.read_parquet(D+'Diagnosis_3.parquet')
uniq  = pd.read_parquet(D+'Unique_Patients_All.parquet')
bio   = pd.read_parquet(D+'Biomarkers.parquet')
fsum  = pd.read_parquet(D+'Final_Summary.parquet')
final = pd.read_parquet(D+'Final_Biomarker_Patients.parquet')

def hr(title):
    print('\n' + '='*90)
    print(title)
    print('='*90)

# ---------------------------------------------------------------------------
hr("1. SHEET SHAPES")
for name, df in [('Diagnosis_1',diag1),('Diagnosis_2',diag2),('Diagnosis_3',diag3),
                  ('Unique_Patients_All',uniq),('Biomarkers',bio),
                  ('Final_Summary',fsum),('Final_Biomarker_Patients',final)]:
    print(f"{name:28s} shape={df.shape}")

# ---------------------------------------------------------------------------
hr("2. DIAGNOSIS_1/2/3 — are these VISIT-level (longitudinal) or patient-level?")
for name, df in [('Diagnosis_1',diag1),('Diagnosis_2',diag2),('Diagnosis_3',diag3)]:
    n_rows = len(df)
    n_rid = df['RID'].nunique()
    print(f"{name}: rows={n_rows}, unique RID={n_rid}, "
          f"=> {n_rows - n_rid} extra visit-rows beyond 1/patient "
          f"({'VISIT-LEVEL, multiple rows/patient' if n_rows>n_rid else 'one row per patient'})")

hr("2b. Does the SAME RID appear in more than one of Diagnosis_1/2/3?")
rid1, rid2, rid3 = set(diag1.RID), set(diag2.RID), set(diag3.RID)
print("RID in both Diagnosis_1 & Diagnosis_2 (CN and MCI at different visits):", len(rid1 & rid2))
print("RID in both Diagnosis_1 & Diagnosis_3 (CN and Dementia at different visits):", len(rid1 & rid3))
print("RID in both Diagnosis_2 & Diagnosis_3 (MCI and Dementia at different visits):", len(rid2 & rid3))
print("RID in all three:", len(rid1 & rid2 & rid3))
print("=> This confirms DIAGNOSIS is PER-VISIT (patients convert/revert over time), "
      "not a fixed patient attribute. This is central to how 'Final_Biomarker_Patients' "
      "must be interpreted: DIAGNOSIS reflects the diagnosis AT THE SELECTED VISIT, not "
      "a lifetime label.")

# ---------------------------------------------------------------------------
hr("3. UNIQUE_PATIENTS_ALL — one row per patient?")
print("rows:", len(uniq), "unique RID:", uniq['RID'].nunique(), "unique PTID:", uniq['PTID'].nunique())
dup_rid = uniq[uniq.duplicated('RID', keep=False)].sort_values('RID')
print("Duplicated RID rows in Unique_Patients_All:", len(dup_rid))
if len(dup_rid):
    print(dup_rid.head(10))
print("\nDiagnosis distribution (Unique_Patients_All):")
print(uniq['GROUP'].value_counts())
print("\nMMSE missing count:", uniq['MMSE_FINAL_SCORE'].isna().sum(), "/", len(uniq))

# ---------------------------------------------------------------------------
hr("4. BIOMARKERS SHEET — visit-level or patient-level? matched on RID only per header")
print("rows:", len(bio), "unique RID:", bio['RID'].nunique(), "unique PTID:", bio['PTID'].nunique())
print("=> header says 'matched on RID only' meaning a patient's biomarker visit was matched to")
print("   ANY diagnosis record for that RID, not necessarily the SAME visit (VISCODE). Checking...")
dup_rid_bio = bio[bio.duplicated('RID', keep=False)]
print("Rows sharing an RID with another row in Biomarkers sheet:", len(dup_rid_bio), " -> patients with >1 biomarker visit:", dup_rid_bio['RID'].nunique())

# ---------------------------------------------------------------------------
hr("5. FINAL_SUMMARY — matched on RID+VISCODE (per header)")
print("rows:", len(fsum), "unique RID:", fsum['RID'].nunique(), "unique (RID,VISCODE):", fsum.drop_duplicates(['RID','VISCODE']).shape[0])
dup_rid_fsum = fsum[fsum.duplicated('RID', keep=False)].sort_values('RID')
print("Rows with a duplicated RID (i.e. patient has >1 visit row in Final_Summary):", len(dup_rid_fsum))
print("Unique patients with >1 row:", dup_rid_fsum['RID'].nunique())
print("\nHow many rows actually HAVE non-null biomarker data (pT217_F not null)?")
print(fsum['pT217_F'].notna().sum(), "out of", len(fsum))

# ---------------------------------------------------------------------------
hr("6. FINAL_BIOMARKER_PATIENTS — the core modeling table. Verify 'one row per patient'.")
print("rows:", len(final))
print("unique RID:", final['RID'].nunique())
print("unique PTID:", final['PTID'].nunique())
dup_rid_final = final[final.duplicated('RID', keep=False)].sort_values('RID')
print("Rows with duplicated RID:", len(dup_rid_final), "| unique RIDs affected:", dup_rid_final['RID'].nunique())
if len(dup_rid_final):
    print(dup_rid_final[['PHASE','PTID','RID','VISCODE','EXAMDATE','DIAGNOSIS','GROUP']].to_string())
dup_ptid_final = final[final.duplicated('PTID', keep=False)].sort_values('PTID')
print("\nRows with duplicated PTID:", len(dup_ptid_final))
full_dup_rows = final[final.duplicated(keep=False)]
print("Fully duplicated rows (all columns identical):", len(full_dup_rows))

hr("6b. Documentation claim: 889 patients, Control 343 (38.6%), MCI 315 (35.4%), Dementia 231 (26.0%)")
vc = final['GROUP'].value_counts()
vc_pct = (final['GROUP'].value_counts(normalize=True)*100).round(1)
print(pd.DataFrame({'count':vc, 'pct':vc_pct}))
print("Total:", len(final))

hr("6c. Documentation claim: 5 phases ADNI1,2,3,4,ADNIGO")
print(final['PHASE'].value_counts())

hr("6d. Datatypes in Final_Biomarker_Patients")
print(final.dtypes)

hr("6e. Missingness in Final_Biomarker_Patients (count + %)")
miss = final.isna().sum()
miss_pct = (final.isna().mean()*100).round(2)
print(pd.DataFrame({'missing_count':miss, 'missing_pct':miss_pct}).sort_values('missing_pct', ascending=False))

hr("6f. -4 sentinel counts per biomarker column (BLoQ marker per docs)")
for col in ['NfL_Q','GFAP_Q','NfL_F','GFAP_F','pT217_F','AB42_F','AB40_F','AB42_AB40_F','pT217_AB42_F']:
    n_neg4 = (final[col] == -4).sum()
    n_neg = (final[col] < 0).sum()
    print(f"{col:15s} == -4: {n_neg4:4d}   | <0 (any negative): {n_neg:4d}   | min={final[col].min()}")

hr("6g. Are -4 values EXCLUSIVE to NfL_F / GFAP_F, or do other cols also have them (doc only mentions NfL_F, GFAP_F)?")
for col in final.select_dtypes(include=[np.number]).columns:
    n_neg4 = (final[col] == -4).sum()
    if n_neg4 > 0:
        print(f"{col}: {n_neg4} rows == -4")

hr("6h. Row-wise: how many biomarkers are -4 or NaN per row? (severity of BLoQ / missingness per patient)")
bio_cols_f = ['pT217_F','AB42_F','AB40_F','AB42_AB40_F','pT217_AB42_F','NfL_F','GFAP_F']
bio_cols_q = ['NfL_Q','GFAP_Q']
tmp = final.copy()
for c in bio_cols_f + bio_cols_q:
    tmp[c+'_bad'] = (tmp[c] == -4) | (tmp[c].isna())
tmp['n_bad_F'] = tmp[[c+'_bad' for c in bio_cols_f]].sum(axis=1)
tmp['n_bad_Q'] = tmp[[c+'_bad' for c in bio_cols_q]].sum(axis=1)
print("Distribution of # bad (missing or -4) Fujirebio fields per patient (of 7):")
print(tmp['n_bad_F'].value_counts().sort_index())
print("\nDistribution of # bad (missing or -4) Quanterix fields per patient (of 2):")
print(tmp['n_bad_Q'].value_counts().sort_index())

hr("6i. Are NfL_Q/GFAP_Q ever missing/-4, or only NfL_F/GFAP_F (as doc implies)?")
print("NfL_Q isna:", final['NfL_Q'].isna().sum(), " == -4:", (final['NfL_Q']==-4).sum())
print("GFAP_Q isna:", final['GFAP_Q'].isna().sum(), " == -4:", (final['GFAP_Q']==-4).sum())
print("NfL_F isna:", final['NfL_F'].isna().sum(), " == -4:", (final['NfL_F']==-4).sum())
print("GFAP_F isna:", final['GFAP_F'].isna().sum(), " == -4:", (final['GFAP_F']==-4).sum())

hr("6j. Complete-case count across ALL 5 'primary' biomarkers (pT217_F, AB42_F, AB40_F, NfL, GFAP) — doc claims 567")
# Need to decide: does 'NfL'/'GFAP' mean _F or _Q or best-available? Test a few definitions.
def complete_mask(nfl_col, gfap_col):
    cols = ['pT217_F','AB42_F','AB40_F', nfl_col, gfap_col]
    m = pd.Series(True, index=final.index)
    for c in cols:
        m &= final[c].notna() & (final[c] != -4)
    return m

m_F = complete_mask('NfL_F','GFAP_F')
m_Q = complete_mask('NfL_Q','GFAP_Q')
print("Complete cases using NfL_F & GFAP_F  :", m_F.sum())
print("Complete cases using NfL_Q & GFAP_Q  :", m_Q.sum())
# coalesce best-available
final['NfL_best'] = final['NfL_Q'].where(final['NfL_Q'].notna() & (final['NfL_Q']!=-4), final['NfL_F'])
final['GFAP_best'] = final['GFAP_Q'].where(final['GFAP_Q'].notna() & (final['GFAP_Q']!=-4), final['GFAP_F'])
m_best = complete_mask('NfL_best','GFAP_best')
# fix complete_mask to allow arbitrary col names already generic - reuse
print("Complete cases using coalesced NfL_best & GFAP_best:", m_best.sum())

hr("6k. Recompute group means for pTau217, AB42, AB40, NfL, GFAP on complete-case (Q-based, since doc numbers look like NfL_Q/GFAP_Q scale) subset, compare to doc Table 5.5")
sub = final[m_Q].copy()
sub['DXLABEL'] = sub['GROUP']
means = sub.groupby('DXLABEL')[['pT217_F','AB42_F','AB40_F','NfL_Q','GFAP_Q']].mean().round(3)
print("N =", len(sub))
print(means)

hr("6l. Same but using NfL_best/GFAP_best coalesced, and using DIAGNOSIS_F complete mask")
sub2 = final[m_best].copy()
means2 = sub2.groupby('GROUP')[['pT217_F','AB42_F','AB40_F','NfL_best','GFAP_best']].mean().round(3)
print("N =", len(sub2))
print(means2)

hr("7. AB42_AB40_F and pT217_AB42_F — verify these are correctly computed ratios (not garbage/leakage columns)")
check = final.dropna(subset=['AB42_F','AB40_F','AB42_AB40_F']).copy()
check = check[(check['AB42_F']!=-4)&(check['AB40_F']!=-4)]
check['recomputed'] = check['AB42_F'] / check['AB40_F']
check['diff'] = (check['recomputed'] - check['AB42_AB40_F']).abs()
print("Max abs diff between AB42_F/AB40_F and stored AB42_AB40_F:", check['diff'].max())
print("Rows with diff > 1e-4:", (check['diff']>1e-4).sum(), "out of", len(check))

check2 = final.dropna(subset=['pT217_F','AB42_F','pT217_AB42_F']).copy()
check2 = check2[(check2['pT217_F']!=-4)&(check2['AB42_F']!=-4)]
check2['recomputed'] = check2['pT217_F'] / check2['AB42_F']
check2['diff'] = (check2['recomputed'] - check2['pT217_AB42_F']).abs()
print("Max abs diff pT217_F/AB42_F vs stored pT217_AB42_F:", check2['diff'].max())
print("Rows with diff > 1e-4:", (check2['diff']>1e-4).sum(), "out of", len(check2))

hr("8. DUPLICATE / CONSTANT / NEAR-CONSTANT COLUMN CHECK on Final_Biomarker_Patients")
for c in final.columns:
    nunique = final[c].nunique(dropna=False)
    print(f"{c:20s} n_unique={nunique}")

hr("9. VISCODE consistency — phase-prefix check")
print(final['VISCODE'].value_counts().head(20))
print("\nDoes VISCODE prefix match PHASE consistently? sample check for ADNI4 rows:")
adni4 = final[final['PHASE']=='ADNI4']
print(adni4['VISCODE'].value_counts())

hr("10. EXAMDATE range & parse check")
print("raw dtype:", final['EXAMDATE'].dtype)
final['EXAMDATE_parsed'] = pd.to_datetime(final['EXAMDATE'], errors='coerce')
n_unparseable = final['EXAMDATE_parsed'].isna().sum() - final['EXAMDATE'].isna().sum()
print("Unparseable EXAMDATE strings:", n_unparseable)
print("min:", final['EXAMDATE_parsed'].min(), "max:", final['EXAMDATE_parsed'].max())
future_dates = final[final['EXAMDATE_parsed'] > pd.Timestamp('2026-07-18')]
print("Rows with EXAMDATE after TODAY (2026-07-18) -- i.e. in-dataset 'future' scheduled visits:", len(future_dates))
if len(future_dates):
    print(future_dates[['PHASE','PTID','VISCODE','EXAMDATE']].head(15).to_string())

hr("11. MMSE range check (should be 0-30)")
print(final['MMSE_FINAL_SCORE'].describe())
print("Out of range (<0 or >30):", ((final['MMSE_FINAL_SCORE']<0)|(final['MMSE_FINAL_SCORE']>30)).sum())
print("MMSE missing:", final['MMSE_FINAL_SCORE'].isna().sum())
print("\nMMSE by group (mean):")
print(final.groupby('GROUP')['MMSE_FINAL_SCORE'].agg(['mean','std','count']))

hr("12. DIAGNOSIS vs GROUP consistency")
print(pd.crosstab(final['DIAGNOSIS'], final['GROUP']))

hr("13. Outlier scan (IQR method) on each numeric biomarker, excluding -4 sentinel and NaN")
num_cols = ['pT217_F','AB42_F','AB40_F','AB42_AB40_F','pT217_AB42_F','NfL_Q','GFAP_Q','NfL_F','GFAP_F','MMSE_FINAL_SCORE']
for c in num_cols:
    s = final[c]
    s = s[(s.notna()) & (s != -4)]
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3-q1
    lo, hi = q1-1.5*iqr, q3+1.5*iqr
    n_out = ((s<lo)|(s>hi)).sum()
    print(f"{c:15s} n={len(s):4d}  min={s.min():10.3f} max={s.max():10.3f}  IQR outliers={n_out:4d} ({100*n_out/len(s):.1f}%)")

hr("14. Cross-check: does Final_Biomarker_Patients == Final_Summary restricted to non-null biomarkers?")
fsum_nonnull = fsum[fsum['pT217_F'].notna()]
print("Final_Summary rows with non-null pT217_F:", len(fsum_nonnull))
print("Final_Biomarker_Patients rows:", len(final))
merged_check = fsum_nonnull.merge(final, on=['RID'], how='outer', indicator=True, suffixes=('_fs','_fb'))
print(merged_check['_merge'].value_counts())

print("\nDONE.")

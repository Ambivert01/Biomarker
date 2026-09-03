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

def hr(t):
    print('\n'+'='*90); print(t); print('='*90)

# -------------------------------------------------------------------------
hr("A. Group means on the F-column complete-case subset (n=567) to match doc Table 5.5")
m_F = pd.Series(True, index=final.index)
for c in ['pT217_F','AB42_F','AB40_F','NfL_F','GFAP_F']:
    m_F &= final[c].notna() & (final[c] != -4)
sub = final[m_F]
print("N =", len(sub))
print(sub.groupby('GROUP')[['pT217_F','AB42_F','AB40_F','NfL_F','GFAP_F']].mean().round(3))
print("\nDoc claims: pTau217 CN=0.176 MCI=0.330 Dem=0.783 | AB42 CN=28.20 MCI=28.17 Dem=28.59")
print("            AB40 CN=328.80 MCI=353.13 Dem=373.37 | NfL CN=22.30 MCI=30.82 Dem=37.05")
print("            GFAP CN=59.66 MCI=87.58 Dem=117.63")

# -------------------------------------------------------------------------
hr("B. NfL_Q / GFAP_Q sentinel breakdown: -4 vs -5 vs other negatives")
for c in ['NfL_Q','GFAP_Q']:
    vc = final.loc[final[c] < 0, c].value_counts()
    print(c, "negative value counts:\n", vc)

# -------------------------------------------------------------------------
hr("C. Platform complementarity: Fuji-clean vs Quanterix-clean vs both, and relation to PHASE")
final2 = final.copy()
final2['fuji_NfL_GFAP_ok'] = (final2['NfL_F']!=-4) & (final2['GFAP_F']!=-4)
final2['quant_NfL_GFAP_ok'] = (final2['NfL_Q']!=-4) & (final2['NfL_Q']!=-5) & (final2['GFAP_Q']!=-4) & (final2['GFAP_Q']!=-5)
ct = pd.crosstab(final2['fuji_NfL_GFAP_ok'], final2['quant_NfL_GFAP_ok'])
print("Rows: Fuji NfL/GFAP OK (T/F) x Cols: Quanterix NfL/GFAP OK (T/F)")
print(ct)
print("\nBy PHASE, % with usable Fuji NfL/GFAP vs Quanterix NfL/GFAP:")
summary = final2.groupby('PHASE').agg(
    n=('RID','count'),
    fuji_ok_pct=('fuji_NfL_GFAP_ok', lambda x: round(100*x.mean(),1)),
    quant_ok_pct=('quant_NfL_GFAP_ok', lambda x: round(100*x.mean(),1)),
)
print(summary)

# -------------------------------------------------------------------------
hr("D. MMSE_FINAL_SCORE == -1 rows -- likely a sentinel, not a true score")
print((final['MMSE_FINAL_SCORE']==-1).sum(), "rows with MMSE == -1")
print(final.loc[final['MMSE_FINAL_SCORE']==-1, ['PHASE','PTID','RID','VISCODE','GROUP','MMSE_FINAL_SCORE']])
print("\nMMSE missingness by group:")
print(final.groupby('GROUP')['MMSE_FINAL_SCORE'].apply(lambda x: f"{x.isna().sum()}/{len(x)} missing ({100*x.isna().mean():.1f}%)"))

# -------------------------------------------------------------------------
hr("E. Is DIAGNOSIS in Final_Biomarker_Patients contemporaneous with the exact biomarker-draw visit?")
# Build the full visit-level diagnosis table (all rows from Diagnosis_1/2/3) and check
# that (RID, VISCODE, DIAGNOSIS) triples in `final` are found verbatim among the raw visit records.
diag_all = pd.concat([diag1.assign(SRC='D1'), diag2.assign(SRC='D2'), diag3.assign(SRC='D3')], ignore_index=True)
key_diag = set(zip(diag_all['RID'], diag_all['VISCODE'], diag_all['DIAGNOSIS']))
final_keys = list(zip(final['RID'], final['VISCODE'], final['DIAGNOSIS']))
found = [k in key_diag for k in final_keys]
print(f"Rows in Final_Biomarker_Patients whose (RID,VISCODE,DIAGNOSIS) triple exists verbatim in the raw visit-level diagnosis records: {sum(found)} / {len(final_keys)}")
missing_idx = [i for i,f in enumerate(found) if not f]
print("Rows NOT found verbatim (potential mismatch / relabeled / VISCODE naming diff):", len(missing_idx))
if missing_idx:
    print(final.iloc[missing_idx][['PHASE','PTID','RID','VISCODE','DIAGNOSIS','GROUP']].head(20))

# -------------------------------------------------------------------------
hr("E2. For patients whose diagnosis changes across visits, which visit did Unique_Patients_All select?")
# Among RIDs appearing in more than one of diag1/2/3 (i.e diagnosis changed over time),
# check whether Unique_Patients_All picked the LAST (most recent EXAMDATE), FIRST, or something else.
rid1, rid2, rid3 = set(diag1.RID), set(diag2.RID), set(diag3.RID)
changers = (rid1&rid2) | (rid1&rid3) | (rid2&rid3)
print("Number of patients with diagnosis change across visits (in raw diagnosis sheets):", len(changers))
changers_in_final = changers & set(final['RID'])
print("Of those, how many ended up in the final 889-patient modeling table:", len(changers_in_final))
# For a sample, check whether the selected visit is the most recent one available for that RID across all 3 diag sheets
diag_all['EXAMDATE'] = pd.to_datetime(diag_all['EXAMDATE'], errors='coerce')
sample_check = []
for rid in list(changers_in_final)[:2000]:
    visits = diag_all[diag_all['RID']==rid].sort_values('EXAMDATE')
    selected = final[final['RID']==rid]
    if len(selected)==0:
        continue
    sel_examdate = pd.to_datetime(selected['EXAMDATE'].values[0])
    last_examdate = visits['EXAMDATE'].max()
    first_examdate = visits['EXAMDATE'].min()
    sample_check.append({
        'RID': rid,
        'selected_examdate': sel_examdate,
        'is_last_visit': sel_examdate == last_examdate,
        'is_first_visit': sel_examdate == first_examdate,
        'n_visits_total': len(visits),
        'n_distinct_diagnoses': visits['DIAGNOSIS'].nunique(),
    })
sc = pd.DataFrame(sample_check)
print(sc[['is_last_visit','is_first_visit']].mean())
print("n_distinct_diagnoses among diagnosis-changers in final set:")
print(sc['n_distinct_diagnoses'].value_counts())

# -------------------------------------------------------------------------
hr("F. Does the biomarker draw EXAMDATE match the diagnosis-visit EXAMDATE exactly (same-day) for all 889?")
diag_all_dates = diag_all[['RID','VISCODE','EXAMDATE']].drop_duplicates()
final_dates = final[['RID','VISCODE','EXAMDATE']].copy()
final_dates['EXAMDATE'] = pd.to_datetime(final_dates['EXAMDATE'])
merged = final_dates.merge(diag_all_dates, on=['RID','VISCODE'], suffixes=('_final','_diag'), how='left')
merged['match'] = merged['EXAMDATE_final'] == merged['EXAMDATE_diag']
print("EXAMDATE matches between final table and raw diagnosis visit record:", merged['match'].sum(), "/", len(merged))
print("Missing diag-side date (no matching RID+VISCODE found at all):", merged['EXAMDATE_diag'].isna().sum())

# -------------------------------------------------------------------------
hr("G. ADNIGO phase - only 3 patients. Check their diagnosis distribution (rare category risk)")
print(final[final['PHASE']=='ADNIGO'][['PHASE','PTID','GROUP','DIAGNOSIS']])

# -------------------------------------------------------------------------
hr("H. VISCODE granularity -- can we derive a numeric 'months since baseline' feature? Check pattern")
print(final['VISCODE'].str.extract(r'(m\d+|y\d+)')[0].value_counts())

print("\nDONE.")

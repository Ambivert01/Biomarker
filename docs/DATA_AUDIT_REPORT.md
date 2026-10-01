# ADNI Plasma Biomarker Dataset — Data Audit & Documentation Verification Report

**Scope:** Every sheet in `ADNI_Diagnosis_Patients__2_.xlsx` was loaded and cross-checked against every
factual claim in `ADNI_ML_Project_Proposal.docx`. This report records what was **confirmed**, what was
**wrong or incomplete in the documentation**, and what **new risks** the audit surfaced that the
documentation does not mention at all. All numbers below are reproducible from
`notebooks/01_data_audit.py`, `02_data_audit_followup.py`, and the assay-agreement script.

---

## 1. Sheet-by-sheet purpose (reverse-engineered from the data, not just the header text)

| Sheet | Grain | Rows | Purpose |
|---|---|---|---|
| `Diagnosis_1/2/3` | **visit-level** | 6,531 / 6,586 / 3,020 | Every visit record where a patient's diagnosis at that visit was scored 1 (CN), 2 (MCI), 3 (Dementia). Same patient (RID) appears many times as they return for follow-up visits. |
| `Unique_Patients_All` | one row / patient (3,700) | 3,700 | One diagnosis record selected per patient (see §3) across all three diagnosis groups, with MMSE attached. |
| `Biomarkers` | visit-level, RID-matched | 2,295 | Plasma biomarker panel results joined to diagnosis **by RID only** (not by visit) — see §4, this is a looser join than `Final_Summary`. |
| `Final_Summary` | one row / patient (3,700) | 3,700 | `Unique_Patients_All` joined to `Biomarkers` **on RID+VISCODE** (same-visit match). Only 889/3,700 rows have non-null biomarkers. |
| `Final_Biomarker_Patients` | one row / patient (889) | 889 | `Final_Summary` filtered to the 889 rows that actually have biomarker values. **This is the modeling table.** |

**Verified true:** `Final_Biomarker_Patients` is genuinely one row per patient — 889 unique `RID`, 889
unique `PTID`, zero duplicate rows, zero duplicate keys. The documentation's claim that "889 rows = 889
unique patients" is **correct**.

**Verified true:** biomarkers and diagnosis in the final table are **contemporaneous** — for all 889
rows, the `(RID, VISCODE, DIAGNOSIS)` triple exists verbatim in the raw visit-level diagnosis sheets, and
the biomarker-draw `EXAMDATE` matches the diagnosis-visit `EXAMDATE` exactly (889/889). So there is no
temporal mismatch between "when the blood was drawn" and "what diagnosis label was attached" — this is a
genuine cross-sectional, same-visit dataset, not a forward-looking or backward-looking label.

---

## 2. Documentation claims verified **correct**

- **889 patients**, Control 343 (38.6%), MCI 315 (35.4%), Dementia 231 (26.0%) — exact match.
- **5 phases**: ADNI1 (55), ADNI2 (102), ADNI3 (161), ADNI4 (568), ADNIGO (3) — exact match.
- **Column dictionary** (§4 of the doc) — every named column exists with the stated dtype/units.
- **AB42/AB40 and pTau217/AB42 ratio columns** are correctly computed from `AB42_F`/`AB40_F`/`pT217_F`
  (verified by recomputation on 886 valid rows; max discrepancy 5e-5, consistent with float rounding).
- **Table 5.5 (mean biomarkers by group)** — reproduced **exactly** on the n=567 complete-case subset
  defined by `pT217_F, AB42_F, AB40_F, NfL_F, GFAP_F` all present and ≠ −4. This confirms the
  documentation's "567 patients" and every mean value in that table used the **Fujirebio** NfL/GFAP
  columns specifically (not Quanterix, not a coalesced value) — the doc doesn't say this explicitly, but
  it's now unambiguous.
- **MMSE range and DIAGNOSIS↔GROUP mapping** are internally consistent (1=Control, 2=MCI, 3=Dementia,
  100% consistent, no label contamination).

## 3. Documentation claims that are **wrong or materially incomplete**

### 3.1 The −4 sentinel is not confined to `NfL_F`/`GFAP_F` as stated
The doc says only `NfL_F` and `GFAP_F` use −4.0 for BLoQ/missing. In fact:
- `AB42_F`: 3 rows = −4.0 (undocumented)
- `AB40_F`: 2 rows = −4.0 (undocumented)
- `AB42_AB40_F`, `pT217_AB42_F`: 3 rows = −4.0 each (propagated from the AB42/AB40 sentinel — confirmed
  these are **not** miscomputed ratios of −4/−4, the source data encodes the sentinel directly in the
  ratio column too)
- `NfL_Q`, `GFAP_Q`: **430 rows = −4.0, plus one row = −5.0 each** — the documentation never mentions
  that the Quanterix columns carry a BLoQ sentinel at all, let alone a second, different sentinel value
  (−5.0). Any pipeline that only special-cases −4.0 in the two Fujirebio columns will silently treat 430
  Quanterix BLoQ readings as real negative concentrations and leave one −5.0 value per column untouched.

**Action taken:** the preprocessing pipeline treats *any* negative value in *any* biomarker column as a
BLoQ/missing sentinel, not just literal −4.0 in the two named columns.

### 3.2 "MMSE available for most patients" understates a large, and *non-random*, missingness problem
MMSE is missing for 355/889 patients (40%) overall — but missingness is wildly uneven by diagnosis:

| Group | MMSE missing |
|---|---|
| Control | 186/343 (**54.2%**) |
| MCI | 127/315 (40.3%) |
| Dementia | 42/231 (**18.2%**) |

This is not missing-completely-at-random. Dementia patients are far more likely to have a recorded MMSE
than Controls — almost certainly a **clinical-workflow artifact** (MMSE is a routine part of a dementia
work-up and less consistently logged for healthy controls), not a biological signal. Left unexamined,
"MMSE present/absent" becomes a shortcut feature that predicts diagnosis through documentation habits
rather than cognition. This is treated as a leakage-adjacent risk in §6 and is explicitly ablated in
training (model with vs. without MMSE, and with vs. without an `MMSE_missing` indicator).

Also found: **2 rows have `MMSE_FINAL_SCORE = -1`**, outside the valid 0–30 range — an unstated sentinel,
not a real score. Recoded to missing.

### 3.3 The two assay platforms are not an incidental "completeness" choice — they are almost perfectly determined by ADNI phase (major confound)

| Phase | n | % with usable **Fujirebio** NfL/GFAP | % with usable **Quanterix** NfL/GFAP |
|---|---|---|---|
| ADNI1 | 55 | 0.0% | 100.0% |
| ADNI2 | 102 | 0.0% | 100.0% |
| ADNI3 | 161 | 0.0% | 99.4% |
| ADNI4 | 568 | 100.0% | 24.3% |
| ADNIGO | 3 | 0.0% | 100.0% |

Fujirebio NfL/GFAP is essentially **only measured in ADNI4**; Quanterix is the only platform used in
ADNI1–3/ADNIGO. This is a structural (protocol-driven) missingness pattern, not a patient-level one.

This matters enormously because **diagnosis prevalence itself is heavily confounded with phase**
(χ² = 241.2, p = 1.3×10⁻⁴⁷):

| Phase | %Control | %MCI | %Dementia |
|---|---|---|---|
| ADNI1 | 3.6% | 12.7% | **83.6%** |
| ADNIGO | 0% | 0% | **100%** |
| ADNI2 | 14.7% | 18.6% | 66.7% |
| ADNI3 | 39.8% | 36.0% | 24.2% |
| ADNI4 | 46.1% | 40.7% | 13.2% |

Early ADNI phases enrolled/retained a case-mix dominated by Dementia; the newest phase (ADNI4, 64% of the
whole dataset) skews heavily toward Control/MCI. **Consequently, "which assay platform has a valid
reading" is a near-perfect proxy for phase, which is itself a near-perfect proxy for diagnosis
prevalence.** A model given raw `NfL_F`/`GFAP_F` with naive missing-value imputation, or given `PHASE` as
a one-hot feature, or given "which platform reported a value" as a feature, has an easy shortcut to
learn cohort identity instead of biology — a shortcut that will not generalize to a new deployment
context with a different phase mix. **This is the single most important finding of this audit** and
directly informs the leakage-control and feature-engineering strategy below (harmonized/coalesced
biomarkers as primary features; phase/platform indicators evaluated only as a controlled ablation, not
used by default).

### 3.4 "Platform selection based on completeness" — the audit ran the actual cross-platform agreement analysis the doc only gestures at
For the 138 patients who happen to have **both** platforms measured validly:

| Biomarker | Pearson r | Spearman r | Mean ratio (F/Q) | Ratio stability across CN/MCI/Dementia |
|---|---|---|---|---|
| NfL (F vs Q) | 0.934 | 0.918 | 1.47× (F reads higher) | 1.449 / 1.485 / 1.488 — stable (±3%) |
| GFAP (F vs Q) | 0.793 | 0.915 | 0.38× (F reads lower) | 0.381 / 0.363 / 0.385 — stable (±6%) |

Both platforms are strongly correlated and the multiplicative offset between them is **stable across
diagnostic groups** — i.e., it is a systematic assay-calibration difference, not a diagnosis-dependent
artifact. This licenses a principled harmonization: convert both platforms onto a common scale (via the
empirical log-space offset) rather than arbitrarily preferring one platform or averaging incompatible raw
scales. This is implemented in feature engineering (§ below) as `NfL_harmonized` / `GFAP_harmonized`.

### 3.5 Scope of the model is "concurrent diagnosis," not "early/prognostic prediction," despite the documentation's framing
`Unique_Patients_All`/`Final_Biomarker_Patients` selects, for each patient, their **most recent** visit
(among visits with diagnosis coded 1/2/3) that also has usable biomarkers (confirmed: for all 209
diagnosis-changing patients who made it into the final table, the selected visit is the last one on
record, 100% of the time). Combined with §1's finding that biomarker draw and diagnosis are same-visit,
this model predicts **the diagnosis a patient has right now, from biomarkers drawn right now** — it does
not predict future conversion from earlier biomarkers. The documentation's opening framing ("early and
accurate diagnosis... interventions most effective in pre-dementia stages") reads like a prognostic
framing; the dataset as constructed supports a **concurrent diagnostic classifier**, which is still
clinically useful (blood test replacing/triaging PET-CSF workups) but is a distinct claim from
"predicts who will develop dementia." This distinction is stated explicitly in the final report so the
model is not over-claimed.

### 3.6 Rare-category risk: `PHASE = ADNIGO` has exactly 3 patients, all Dementia
Any encoding of `PHASE` as a plain categorical/one-hot risks the model memorizing these 3 rows outright,
and stratified CV can behave unpredictably with a 3-row category. ADNIGO is grouped into the nearest
adjacent cohort (`ADNI2/3`, the phases immediately bracketing ADNIGO enrollment) or dropped from
phase-based features entirely, and PHASE is treated as an ablation-only covariate per §3.3.

---

## 4. Full integrity checklist (per the project brief)

| Check | Result |
|---|---|
| Duplicate rows in `Final_Biomarker_Patients` | 0 |
| Duplicate RID / PTID | 0 / 0 |
| Duplicate VISCODE (within patient) | N/A — one row/patient |
| Patient leakage across sheets | `Final_Biomarker_Patients` itself has no patient overlap issue (889 unique RID). Cross-sheet: 747 patients change diagnosis across visits in the raw longitudinal sheets; only the single most-recent, biomarker-complete visit per patient enters the modeling table, so **no single patient contributes more than one row to the modeling table** — safe for the planned patient-level stratified splits. |
| Null values | Only `MMSE_FINAL_SCORE` shows as null via `.isna()` (355/889, 40%, uneven by group — see §3.2). All other "missingness" is hidden inside negative sentinels (§3.1), which `.isna()` does not catch. |
| Invalid / impossible values | Negative "concentrations" (physically impossible) confirmed in `NfL_Q`, `GFAP_Q` (−4, −5), `NfL_F`, `GFAP_F` (−4), `AB42_F`, `AB40_F` (−4) and their derived ratios — all sentinel codes, not real measurements, recoded to missing. `MMSE = -1` (2 rows) — recoded to missing. |
| Outliers (IQR) | 1.5%–5.6% per biomarker after removing sentinels; `GFAP_F` max = 2,478 pg/mL is a genuine extreme value (not a sentinel) — retained but handled with robust scaling / winsorization option, not deleted, since extreme GFAP is clinically plausible in advanced neurodegeneration. |
| Constant / near-constant columns | None constant. `PHASE=ADNIGO` (n=3) is a near-constant / rare category (§3.6). |
| Data types | `EXAMDATE` and `VISCODE` are stored as **strings**, not native dates — required explicit `pd.to_datetime` parsing (0 unparseable after conversion, range 2007-03-09 to 2026-05-08, no future-dated anomalies). |
| String inconsistencies | `VISCODE` naming is phase-dependent (`4_bl`, `4_m12` vs `bl`, `m12`, `v11`, `y1`...) but internally consistent per phase — no typos/case issues found. |
| Phase inconsistencies | VISCODE prefixes agree with PHASE in every sampled case. |
| Target distribution | Mild imbalance as documented: 38.6% / 35.4% / 26.0% — not severe enough to require synthetic oversampling by itself, but combined with the phase confound (§3.3) the *effective* per-phase class balance is much more skewed, so `class_weight='balanced'` and stratified (by both GROUP and, in sensitivity runs, PHASE) splitting are used. |

---

## 5. Consequences for pipeline design (decisions made because of this audit)

1. **Sentinel handling is generalized**, not hard-coded to `-4.0` in two columns: any biomarker value
   `< 0` (covers -4 and the undocumented -5) across *all* biomarker columns is recoded to `NaN` before
   any imputation, scaling, or ratio use.
2. **Cross-platform harmonization** for NfL and GFAP: rather than picking one platform or leaving two
   parallel half-empty columns, a `*_harmonized` feature is engineered using the stable, group-invariant
   log-space calibration offset from §3.4, coalesced with whichever platform is actually available per
   patient. Both raw platform columns are *also* kept for the model-comparison ablation, but the
   harmonized versions are the default engineered feature.
3. **PHASE and platform-availability are excluded from the default feature set** and only reintroduced in
   a controlled ablation run, specifically to measure (and report) how much apparent accuracy is
   attributable to cohort/protocol shortcuts rather than biomarker biology — directly answering the
   brief's "assay analysis" and "data leakage" requirements with evidence rather than assumption.
4. **MMSE is modeled two ways** (with vs. without, and with vs. without a missingness indicator) because
   of the informative-missingness pattern in §3.2, and the final choice is justified by nested
   cross-validated performance plus an explicit discussion of the shortcut-learning risk, not by
   accuracy alone.
5. **All splits are patient-level** (safe by construction here, since the modeling table is already
   one-row-per-patient) and **stratified by diagnosis**; a phase-stratified sensitivity split is run
   separately to test robustness to the cohort confound.
6. **Log-transforms** are used for `pT217_F`, `AB40_F`, `NfL_*`, `GFAP_*` (raw skewness 3.5–18.2, reduced
   to <1 after `log1p`), consistent with the brief's feature-engineering request and justified
   empirically rather than by default.

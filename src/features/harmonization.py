"""
harmonization.py
=================
Addresses DATA_AUDIT_REPORT.md §3.3/§3.4: NfL and GFAP are each measured on
two assay platforms (Fujirebio '_F' and Quanterix '_Q') that are almost
perfectly determined by ADNI phase, and phase is itself heavily confounded
with diagnosis (chi2 p=1.3e-47). Naively picking one platform, or leaving
two parallel half-empty columns for a model to impute, risks the model
learning "which cohort/phase this patient came from" as a shortcut for
"which platform reported a number" instead of learning the biology.

Per the audit, for the 138 patients with BOTH platforms valid, the two
platforms are strongly correlated (Pearson r=0.93 NfL, r=0.79 GFAP) with a
STABLE multiplicative offset across diagnostic groups (<=6% relative
variation CN vs MCI vs Dementia) — i.e. this is a systematic assay
calibration difference, not a diagnosis-dependent artifact, so it is
statistically sound to bridge the two platforms onto one common scale.

IMPORTANT (leakage control): the calibration ratio is NOT a hardcoded global
constant computed once on the full 889-patient dataset (which would leak
test-set feature information into training, even though it does not touch
the target). It is *fit* on whichever rows are visible at `.fit()` time
(the training fold only) and reused unchanged for validation/test/inference,
exactly like any other sklearn transformer statistic (mean, scale, etc).
A config-supplied fallback constant (audited on the full 889-patient set) is
used only in the edge case where a given training fold happens to contain
too few dual-platform patients to estimate the ratio reliably.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CrossPlatformHarmonizer(BaseEstimator, TransformerMixin):
    """
    Produces harmonized NfL/GFAP features by converting Quanterix readings
    onto the Fujirebio scale (arbitrary choice of reference platform; ADNI4,
    the largest and most recent cohort, reports Fujirebio) and coalescing
    with whatever platform is actually available per patient.

    New columns added: {analyte}_harmonized, {analyte}_n_platforms (0/1/2 —
    diagnostic only, excluded from the default model feature set, see
    config `models_to_benchmark` feature-set ablation).
    """

    def __init__(self, fallback_nfl_ratio: float = 1.465, fallback_gfap_ratio: float = 0.376,
                 min_pairs_for_fit: int = 15):
        self.fallback_nfl_ratio = fallback_nfl_ratio
        self.fallback_gfap_ratio = fallback_gfap_ratio
        self.min_pairs_for_fit = min_pairs_for_fit

    def _estimate_ratio(self, f_col: pd.Series, q_col: pd.Series, fallback: float) -> float:
        both = f_col.notna() & q_col.notna()
        n_pairs = both.sum()
        if n_pairs < self.min_pairs_for_fit:
            return fallback
        ratio = (f_col[both] / q_col[both]).median()
        if not np.isfinite(ratio) or ratio <= 0:
            return fallback
        return float(ratio)

    def fit(self, X: pd.DataFrame, y=None):
        self.nfl_ratio_ = self._estimate_ratio(X["NfL_F"], X["NfL_Q"], self.fallback_nfl_ratio)
        self.gfap_ratio_ = self._estimate_ratio(X["GFAP_F"], X["GFAP_Q"], self.fallback_gfap_ratio)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # --- NfL ---
        nfl_q_as_f = X["NfL_Q"] * self.nfl_ratio_
        X["NfL_n_platforms"] = X["NfL_F"].notna().astype(int) + X["NfL_Q"].notna().astype(int)
        X["NfL_harmonized"] = np.where(
            X["NfL_F"].notna() & X["NfL_Q"].notna(), (X["NfL_F"] + nfl_q_as_f) / 2.0,
            np.where(X["NfL_F"].notna(), X["NfL_F"],
                     np.where(X["NfL_Q"].notna(), nfl_q_as_f, np.nan)))
        # --- GFAP ---
        gfap_q_as_f = X["GFAP_Q"] * self.gfap_ratio_
        X["GFAP_n_platforms"] = X["GFAP_F"].notna().astype(int) + X["GFAP_Q"].notna().astype(int)
        X["GFAP_harmonized"] = np.where(
            X["GFAP_F"].notna() & X["GFAP_Q"].notna(), (X["GFAP_F"] + gfap_q_as_f) / 2.0,
            np.where(X["GFAP_F"].notna(), X["GFAP_F"],
                     np.where(X["GFAP_Q"].notna(), gfap_q_as_f, np.nan)))
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.asarray(base + ["NfL_harmonized", "NfL_n_platforms", "GFAP_harmonized", "GFAP_n_platforms"])

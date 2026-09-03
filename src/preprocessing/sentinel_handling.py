"""
sentinel_handling.py
=====================
Converts ADNI's BLoQ / missing sentinel codes into proper NaN values,
BEFORE any imputation, scaling, or ratio computation happens.

Per DATA_AUDIT_REPORT.md §3.1: the documentation only mentions -4.0 in
NfL_F/GFAP_F, but the audit found -4.0 in AB42_F/AB40_F/ratios too, and an
undocumented -5.0 in NfL_Q/GFAP_Q. We therefore treat ANY negative value in
ANY biomarker column as a sentinel, rather than hardcoding -4.0 in two
named columns. This is deliberately generic so it is robust to future
sentinel codes that might appear in a data refresh.

This is a stateless transformer (no fit-time statistics), so it is safe to
apply identically to train/val/test/inference without any leakage risk.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SentinelToNaN(BaseEstimator, TransformerMixin):
    """
    Recode sentinel / physically-impossible values to NaN.

    Parameters
    ----------
    biomarker_columns : list[str]
        Columns where "negative concentration" is physically impossible and
        therefore always a sentinel code (BLoQ / not-run / data-entry code).
    negative_threshold : float
        Values strictly below this are recoded to NaN. Default 0.0.
    mmse_column : str | None
        If given, out-of-range MMSE values are also recoded.
    mmse_valid_range : tuple[float, float]
        Inclusive valid range for MMSE.
    """

    def __init__(self, biomarker_columns, negative_threshold: float = 0.0,
                 mmse_column: str | None = None, mmse_valid_range=(0, 30)):
        self.biomarker_columns = biomarker_columns
        self.negative_threshold = negative_threshold
        self.mmse_column = mmse_column
        self.mmse_valid_range = mmse_valid_range

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless — nothing to learn from data (prevents any leakage).
        # `fitted_` exists only so sklearn's check_is_fitted() recognizes
        # this transformer as fitted; it carries no learned information.
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.biomarker_columns:
            if col in X.columns:
                mask = X[col] < self.negative_threshold
                n_flagged = mask.sum()
                if n_flagged:
                    X.loc[mask, col] = np.nan
        if self.mmse_column is not None and self.mmse_column in X.columns:
            lo, hi = self.mmse_valid_range
            bad = (X[self.mmse_column] < lo) | (X[self.mmse_column] > hi)
            X.loc[bad, self.mmse_column] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)

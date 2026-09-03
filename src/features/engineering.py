"""
engineering.py
===============
Clinically-motivated feature engineering, applied AFTER sentinel cleaning
and cross-platform harmonization. All transforms here are either (a)
stateless deterministic functions of a row's own values (ratios, logs,
diffs -> zero leakage risk), or (b) fit only on the training fold
(rank/z-score/quantile-bin edges -> fit() stores training-fold statistics,
transform() applies them unchanged to val/test/inference).

Feature families implemented (see PROJECT_REPORT.md for the ablation that
decided which ones are kept in the final feature set):
  - AB42/AB40 ratio (already provided as AB42_AB40_F, kept + recomputed from
    harmonized inputs is NOT applicable since AB42/AB40 are Fuji-only, no Q
    equivalent exists in this dataset)
  - pTau217/AB42 ratio (already provided as pT217_AB42_F)
  - GFAP/NfL ratio (harmonized) — neuroinflammation-to-injury ratio, not in
    source data, clinically motivated (astrogliosis vs axonal injury balance)
  - AB40 - AB42 (absolute amyloid deposition proxy)
  - log1p transforms of right-skewed biomarkers (skew 3.5-18.2 raw, see audit §M)
  - z-score transforms (fit on train fold only, via the ColumnTransformer's
    StandardScaler downstream — NOT duplicated here to avoid redundant leakage
    surface; this module only adds the rank/quantile-bin features which
    StandardScaler does not provide)
  - rank transform (percentile rank of each biomarker within the training
    distribution — robust to outliers, fit on train fold only)
  - MMSE inclusion is a *set-level* switch (config flag), not a per-row
    transform, tested in the ablation in training/train_pipeline.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


LOG_TRANSFORM_COLS = ["pT217_F", "AB40_F", "NfL_harmonized", "GFAP_harmonized"]
# AB42_F excluded from log-transform list: audit found log1p makes AB42_F skew
# *more* negative (-0.305) without materially helping separation; raw AB42_F
# already has low skew relative to the others (2.88 vs 6-18 for the rest).


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds engineered ratio/log/rank features. `include_rank_features` and
    `include_interaction_features` are ablation switches so the training
    script can empirically test "does this feature family help?" instead of
    including everything by assumption (per project brief: "only retain
    features that improve validation").
    """

    def __init__(self, include_rank_features: bool = True,
                 include_interaction_features: bool = True,
                 include_log_features: bool = True):
        self.include_rank_features = include_rank_features
        self.include_interaction_features = include_interaction_features
        self.include_log_features = include_log_features

    def fit(self, X: pd.DataFrame, y=None):
        # Rank transform reference distributions are learned on the training
        # fold ONLY and frozen; new data (val/test/inference) is scored
        # against this frozen training distribution via searchsorted, so a
        # single new patient does not shift anyone else's rank (no leakage,
        # and stable for single-row production inference).
        self._rank_refs_ = {}
        if self.include_rank_features:
            for col in ["pT217_F", "AB42_F", "AB40_F", "NfL_harmonized", "GFAP_harmonized"]:
                if col in X.columns:
                    vals = X[col].dropna().sort_values().to_numpy()
                    self._rank_refs_[col] = vals
        return self

    def _percentile_rank(self, col: str, series: pd.Series) -> pd.Series:
        ref = self._rank_refs_.get(col)
        if ref is None or len(ref) == 0:
            return pd.Series(np.nan, index=series.index)
        idx = np.searchsorted(ref, series.to_numpy(), side="right")
        pct = idx / len(ref)
        pct = np.where(series.isna(), np.nan, pct)
        return pd.Series(pct, index=series.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # --- Ratios (already present in source, kept as-is; recompute a
        #     couple of derived ones not present in source) ---
        if self.include_interaction_features:
            if "GFAP_harmonized" in X.columns and "NfL_harmonized" in X.columns:
                X["GFAP_NfL_ratio"] = X["GFAP_harmonized"] / X["NfL_harmonized"].replace(0, np.nan)
            if "AB40_F" in X.columns and "AB42_F" in X.columns:
                X["AB40_minus_AB42"] = X["AB40_F"] - X["AB42_F"]
            if "pT217_F" in X.columns and "GFAP_harmonized" in X.columns:
                X["pT217_GFAP_product"] = X["pT217_F"] * X["GFAP_harmonized"] / 100.0  # scaled to keep magnitude sane

        # --- log1p transforms of skewed biomarkers ---
        if self.include_log_features:
            for col in LOG_TRANSFORM_COLS:
                if col in X.columns:
                    safe = X[col].clip(lower=0)
                    X[f"log_{col}"] = np.log1p(safe)

        # --- percentile rank transforms (outlier-robust) ---
        if self.include_rank_features:
            for col in ["pT217_F", "AB42_F", "AB40_F", "NfL_harmonized", "GFAP_harmonized"]:
                if col in X.columns:
                    X[f"rank_{col}"] = self._percentile_rank(col, X[col])

        return X

    def get_feature_names_out(self, input_features=None):
        # Not exhaustively enumerated; ColumnTransformer downstream selects
        # explicit column lists rather than relying on this for correctness.
        return None

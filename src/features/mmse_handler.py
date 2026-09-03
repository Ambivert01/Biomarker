"""
mmse_handler.py
================
Per DATA_AUDIT_REPORT.md §3.2: MMSE missingness is NOT random — it is
54.2% missing for Control, 40.3% for MCI, 18.2% for Dementia. This is almost
certainly a clinical-documentation artifact (MMSE is a routine part of a
dementia work-up, less consistently logged for healthy controls) rather
than a biological signal. An "MMSE recorded / not recorded" indicator is
therefore a *shortcut* feature that can inflate apparent accuracy without
reflecting true diagnostic biology, and it will not generalize to a
deployment setting with different documentation habits.

This transformer always emits an explicit `MMSE_missing` flag so the effect
can be measured directly, but whether that flag (and MMSE itself) is
actually included in the model's feature set is a config-level ablation
switch decided empirically in training/train_pipeline.py — not baked in
here as an assumption.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MMSEHandler(BaseEstimator, TransformerMixin):
    def __init__(self, mmse_column: str = "MMSE_FINAL_SCORE"):
        self.mmse_column = mmse_column

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True  # stateless; marker only, for sklearn's check_is_fitted()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.mmse_column in X.columns:
            X["MMSE_missing"] = X[self.mmse_column].isna().astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.asarray(base + ["MMSE_missing"])

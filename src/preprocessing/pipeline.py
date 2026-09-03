"""
pipeline.py
============
Assembles the full leakage-safe preprocessing pipeline:

  raw DataFrame
    -> SentinelToNaN                 (stateless: -4/-5/negatives -> NaN)
    -> CrossPlatformHarmonizer       (fit on train fold: NfL/GFAP platform bridging)
    -> ClinicalFeatureEngineer       (fit on train fold: ratios, logs, ranks)
    -> MMSEHandler                   (stateless: missingness flag)
    -> FeatureSetSelector            (config-driven column selection = ablation switch)
    -> ColumnTransformer             (fit on train fold: impute + scale + encode)
    -> numpy array ready for a classifier

Every step with any fitted statistic (ratios, rank references, imputer
values, scaler mean/scale) is a proper sklearn Transformer fit only inside
`Pipeline.fit()`, so wrapping this whole object in cross_val_score /
GridSearchCV / a manual CV loop is automatically leakage-safe: fit() is
called fresh on each training fold and transform() on the held-out fold.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

from preprocessing.sentinel_handling import SentinelToNaN
from features.harmonization import CrossPlatformHarmonizer
from features.engineering import ClinicalFeatureEngineer
from features.mmse_handler import MMSEHandler


class FeatureSetSelector(BaseEstimator, TransformerMixin):
    """Config-driven ablation switch: pick which engineered column groups survive into the model."""

    def __init__(self, numeric_features: list[str], categorical_features: list[str] | None = None):
        # sklearn convention: store constructor params UNMODIFIED (needed for
        # clone() round-tripping inside cross_validate/GridSearchCV/Optuna).
        # `None` -> `[]` defaulting happens in transform(), not here.
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        self.fitted_ = True  # stateless; marker only, for sklearn's check_is_fitted()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self.numeric_features + (self.categorical_features or [])
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"FeatureSetSelector: expected columns not found after feature engineering: {missing}")
        return X[cols]


def get_imputer(strategy: str, knn_neighbors: int = 5, random_state: int = 42):
    if strategy == "median":
        return SimpleImputer(strategy="median")
    if strategy == "mean":
        return SimpleImputer(strategy="mean")
    if strategy == "knn":
        return KNNImputer(n_neighbors=knn_neighbors)
    if strategy == "iterative":
        return IterativeImputer(random_state=random_state, max_iter=15, sample_posterior=False)
    raise ValueError(f"Unknown imputation strategy: {strategy}")


def get_scaler(strategy: str):
    if strategy == "standard":
        return StandardScaler()
    if strategy == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown scaling strategy: {strategy}")


DEFAULT_SENTINEL_COLUMNS = [
    "pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F",
    "NfL_Q", "GFAP_Q", "NfL_F", "GFAP_F",
]


def build_feature_engineering_steps(include_rank_features=True, include_interaction_features=True,
                                     include_log_features=True, mmse_column="MMSE_FINAL_SCORE",
                                     harmonizer_kwargs=None, sentinel_kwargs=None):
    """Everything up to (not including) column selection / impute / scale."""
    harmonizer_kwargs = harmonizer_kwargs or {}
    sentinel_kwargs = sentinel_kwargs or {}
    sentinel_kwargs.setdefault("biomarker_columns", DEFAULT_SENTINEL_COLUMNS)
    sentinel_kwargs.setdefault("mmse_column", mmse_column)
    return Pipeline(steps=[
        ("sentinel_to_nan", SentinelToNaN(**sentinel_kwargs)),
        ("harmonize_platforms", CrossPlatformHarmonizer(**harmonizer_kwargs)),
        ("engineer_features", ClinicalFeatureEngineer(
            include_rank_features=include_rank_features,
            include_interaction_features=include_interaction_features,
            include_log_features=include_log_features)),
        ("mmse_handler", MMSEHandler(mmse_column=mmse_column)),
    ])


def build_full_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
    imputation_strategy: str = "median",
    scaling_strategy: str = "robust",
    knn_neighbors: int = 5,
    random_state: int = 42,
    feature_engineering_kwargs: dict | None = None,
) -> Pipeline:
    """
    Returns a single sklearn Pipeline: raw DataFrame in -> numpy array out.
    Safe to drop directly into cross_val_score, GridSearchCV, Optuna
    objectives, or Pipeline([... , ('clf', model)]).
    """
    categorical_features = categorical_features or []
    fe_kwargs = feature_engineering_kwargs or {}

    fe_steps = build_feature_engineering_steps(**fe_kwargs)

    numeric_transformer = Pipeline(steps=[
        ("impute", get_imputer(imputation_strategy, knn_neighbors, random_state)),
        ("scale", get_scaler(scaling_strategy)),
    ])

    transformers = [("numeric", numeric_transformer, numeric_features)]
    if categorical_features:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features))

    column_transform = ColumnTransformer(transformers=transformers, remainder="drop")

    full_pipeline = Pipeline(steps=[
        ("feature_engineering", fe_steps),
        ("select_features", FeatureSetSelector(numeric_features, categorical_features)),
        ("column_transform", column_transform),
    ])
    return full_pipeline


# ---------------------------------------------------------------------------
# Canonical feature-set definitions used across the project (kept in one
# place so training / evaluation / inference can never drift out of sync).
# ---------------------------------------------------------------------------

CORE_BIOMARKER_FEATURES = [
    "pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F",
    "NfL_harmonized", "GFAP_harmonized",
]

ENGINEERED_FEATURES = [
    "GFAP_NfL_ratio", "AB40_minus_AB42", "pT217_GFAP_product",
    "log_pT217_F", "log_AB40_F", "log_NfL_harmonized", "log_GFAP_harmonized",
    "rank_pT217_F", "rank_AB42_F", "rank_AB40_F", "rank_NfL_harmonized", "rank_GFAP_harmonized",
]

MMSE_FEATURES = ["MMSE_FINAL_SCORE", "MMSE_missing"]

# Deliberately NOT in any default set (see audit §3.3): PHASE / platform counts
CONFOUND_RISK_FEATURES = ["NfL_n_platforms", "GFAP_n_platforms"]
PHASE_CATEGORICAL = ["PHASE"]


def get_feature_set(name: str) -> list[str]:
    """
    Named, reproducible feature-set configurations used by the ablation study
    in training/train_pipeline.py.
    """
    sets = {
        "core_only": CORE_BIOMARKER_FEATURES,
        "core_plus_engineered": CORE_BIOMARKER_FEATURES + ENGINEERED_FEATURES,
        "core_plus_engineered_plus_mmse": CORE_BIOMARKER_FEATURES + ENGINEERED_FEATURES + MMSE_FEATURES,
        "core_plus_mmse_no_engineered": CORE_BIOMARKER_FEATURES + MMSE_FEATURES,
        "everything_incl_confound_risk": CORE_BIOMARKER_FEATURES + ENGINEERED_FEATURES + MMSE_FEATURES + CONFOUND_RISK_FEATURES,
    }
    if name not in sets:
        raise ValueError(f"Unknown feature set '{name}'. Options: {list(sets)}")
    return sets[name]

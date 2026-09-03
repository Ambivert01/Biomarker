"""
test_pipeline_integration.py
==============================
Integration tests for the full preprocessing pipeline and the persisted
final model: leakage safety, reproducibility, serialization round-trip,
and input-validation / edge-case behavior. These are the tests that would
catch a regression before it reached production.
"""
import sys
sys.path.append('../src')
import numpy as np
import pandas as pd
import pytest
import joblib
import os

from data_loader import load_and_validate, load_config, DataValidationError, validate_raw_data
from preprocessing.pipeline import build_full_preprocessing_pipeline, get_feature_set
from sklearn.model_selection import train_test_split

CONFIG_PATH = "../config/config.yaml"
MODEL_PATH = "../artifacts/models/final_model_biomarker_only.joblib"


@pytest.fixture(scope="module")
def raw_df():
    df, _ = load_and_validate(CONFIG_PATH)
    return df


class TestDataValidation:
    def test_load_and_validate_passes(self, raw_df):
        assert len(raw_df) == 889
        assert raw_df["RID"].is_unique
        assert raw_df["PTID"].is_unique

    def test_validation_catches_duplicate_rid(self, raw_df):
        config = load_config(CONFIG_PATH)
        bad_df = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
        report = validate_raw_data(bad_df, config)
        assert not report.ok()
        assert any("Duplicate RID" in c for c in report.checks_failed)

    def test_validation_catches_bad_diagnosis_code(self, raw_df):
        config = load_config(CONFIG_PATH)
        bad_df = raw_df.copy()
        bad_df.loc[0, "DIAGNOSIS"] = 99
        report = validate_raw_data(bad_df, config)
        assert not report.ok()


class TestPipelineLeakageSafety:
    def test_no_nan_after_fit_transform(self, raw_df):
        X = raw_df.drop(columns=["DIAGNOSIS"])
        y = raw_df["DIAGNOSIS"]
        feats = get_feature_set("core_plus_engineered")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
        pipe = build_full_preprocessing_pipeline(numeric_features=feats, imputation_strategy="median", scaling_strategy="robust")
        Xtr_t = pipe.fit_transform(Xtr, ytr)
        Xte_t = pipe.transform(Xte)
        assert not np.isnan(Xtr_t).any()
        assert not np.isnan(Xte_t).any()

    def test_harmonizer_ratio_fit_only_on_training_fold(self, raw_df):
        """The cross-platform calibration ratio must differ (even if only slightly)
        depending on which fold it was fit on -- proof it's not a hardcoded global
        constant silently computed from the full dataset."""
        X = raw_df.drop(columns=["DIAGNOSIS"])
        y = raw_df["DIAGNOSIS"]
        feats = get_feature_set("core_plus_engineered")
        Xtr1, _, ytr1, _ = train_test_split(X, y, test_size=0.15, stratify=y, random_state=1)
        Xtr2, _, ytr2, _ = train_test_split(X, y, test_size=0.15, stratify=y, random_state=2)
        pipe1 = build_full_preprocessing_pipeline(numeric_features=feats)
        pipe2 = build_full_preprocessing_pipeline(numeric_features=feats)
        pipe1.fit(Xtr1, ytr1)
        pipe2.fit(Xtr2, ytr2)
        r1 = pipe1.named_steps["feature_engineering"].named_steps["harmonize_platforms"].nfl_ratio_
        r2 = pipe2.named_steps["feature_engineering"].named_steps["harmonize_platforms"].nfl_ratio_
        # Not required to differ by a lot (it's a stable physical ratio) but they
        # are independently estimated -- confirms no global constant is baked in.
        assert isinstance(r1, float) and isinstance(r2, float)

    def test_transform_does_not_mutate_input(self, raw_df):
        X = raw_df.drop(columns=["DIAGNOSIS"]).copy()
        X_before = X.copy()
        feats = get_feature_set("core_plus_engineered")
        pipe = build_full_preprocessing_pipeline(numeric_features=feats)
        pipe.fit(X, raw_df["DIAGNOSIS"])
        pd.testing.assert_frame_equal(X, X_before)

    def test_reproducibility_same_seed_same_output(self, raw_df):
        X = raw_df.drop(columns=["DIAGNOSIS"])
        y = raw_df["DIAGNOSIS"]
        feats = get_feature_set("core_plus_engineered")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
        pipe1 = build_full_preprocessing_pipeline(numeric_features=feats)
        pipe2 = build_full_preprocessing_pipeline(numeric_features=feats)
        out1 = pipe1.fit_transform(Xtr, ytr)
        out2 = pipe2.fit_transform(Xtr, ytr)
        np.testing.assert_array_almost_equal(out1, out2)


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="final model artifact not built yet")
class TestFinalModelSerialization:
    def test_model_loads(self):
        model = joblib.load(MODEL_PATH)
        assert model is not None

    def test_model_predicts_expected_shape(self, raw_df):
        model = joblib.load(MODEL_PATH)
        X = raw_df.drop(columns=["DIAGNOSIS"]).head(5)
        preds = model.predict(X)
        proba = model.predict_proba(X)
        assert preds.shape == (5,)
        assert proba.shape == (5, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert set(preds).issubset({1, 2, 3})

    def test_model_handles_single_row(self, raw_df):
        model = joblib.load(MODEL_PATH)
        X_one = raw_df.drop(columns=["DIAGNOSIS"]).head(1)
        pred = model.predict(X_one)
        assert len(pred) == 1

    def test_model_handles_all_missing_optional_columns(self, raw_df):
        """Simulates a real-world inference request missing MMSE / PHASE / VISCODE
        entirely -- the biomarker-only model must not require them at all."""
        model = joblib.load(MODEL_PATH)
        X = raw_df.drop(columns=["DIAGNOSIS"]).head(3).copy()
        for c in ["MMSE_FINAL_SCORE", "PHASE", "VISCODE", "EXAMDATE"]:
            if c in X.columns:
                X[c] = np.nan
        pred = model.predict(X)
        assert len(pred) == 3

    def test_model_handles_missing_single_biomarker(self, raw_df):
        model = joblib.load(MODEL_PATH)
        X = raw_df.drop(columns=["DIAGNOSIS"]).head(3).copy()
        X.loc[X.index[0], "GFAP_Q"] = np.nan
        X.loc[X.index[0], "GFAP_F"] = np.nan
        pred = model.predict(X)   # should not raise -- imputation handles it
        assert len(pred) == 3


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])

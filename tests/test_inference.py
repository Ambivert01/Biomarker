"""
test_inference.py
===================
Tests the production inference pipeline's input validation and the edge
cases called out in the project brief (missing biomarkers, NaN, negative
values, zero values, unknown/invalid datatypes, empty input, outliers).
Covers BOTH deployed models: binary (Control vs Dementia, recommended) and
three_class (Control vs MCI vs Dementia, reference).
"""
import sys, os
sys.path.append('../src')
import numpy as np
import pytest

from api.inference import ADNIInferencePipeline, InputValidationError, THRESHOLD_MODES

MODEL_DIR = "../artifacts/models"


@pytest.fixture(scope="module", params=["binary", "three_class"])
def pipeline(request):
    mode = request.param
    cfg_files = {
        "binary": "final_model_binary.joblib",
        "three_class": "final_model_biomarker_only.joblib",
    }
    if not os.path.exists(f"{MODEL_DIR}/{cfg_files[mode]}"):
        pytest.skip(f"{mode} model artifact not built yet")
    return ADNIInferencePipeline(mode=mode, model_dir=MODEL_DIR)


VALID_RECORD = {
    "pT217_F": 0.45, "AB42_F": 27.0, "AB40_F": 340.0,
    "AB42_AB40_F": 0.079, "pT217_AB42_F": 0.017,
    "NfL_F": 30.0, "GFAP_F": 75.0, "NfL_Q": None, "GFAP_Q": None,
}


class TestValidInference:
    def test_valid_record_returns_prediction(self, pipeline):
        result = pipeline.predict_one(VALID_RECORD, threshold_mode="balanced", explain=True)
        assert result.diagnosis in ["Control", "MCI", "Dementia"]
        assert 0.0 <= result.confidence <= 1.0
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6

    def test_all_three_threshold_modes_work(self, pipeline):
        for mode in THRESHOLD_MODES[pipeline.mode]:
            result = pipeline.predict_one(VALID_RECORD, threshold_mode=mode, explain=False)
            assert result.diagnosis_code in [1, 2, 3]

    def test_json_serializable(self, pipeline):
        result = pipeline.predict_one(VALID_RECORD, explain=True)
        s = result.to_json()
        assert '"diagnosis"' in s

    def test_model_accuracy_included_in_output(self, pipeline):
        """Explicit requirement: every prediction must carry the model's validated accuracy."""
        result = pipeline.predict_one(VALID_RECORD, explain=False)
        assert "point_estimate" in result.model_accuracy
        assert "ci_95_low" in result.model_accuracy
        assert "ci_95_high" in result.model_accuracy
        assert 0.0 <= result.model_accuracy["point_estimate"] <= 1.0
        assert result.model_accuracy["ci_95_low"] <= result.model_accuracy["point_estimate"] <= result.model_accuracy["ci_95_high"]

    def test_binary_model_never_predicts_mci(self):
        if not os.path.exists(f"{MODEL_DIR}/final_model_binary.joblib"):
            pytest.skip("binary model artifact not built yet")
        pipe = ADNIInferencePipeline(mode="binary", model_dir=MODEL_DIR)
        result = pipe.predict_one(VALID_RECORD, explain=False)
        assert result.diagnosis != "MCI"
        assert result.diagnosis_code != 2

    def test_binary_model_accuracy_clears_80_percent(self):
        """The whole point of building the binary model: verify its validated accuracy is >= 0.80."""
        if not os.path.exists(f"{MODEL_DIR}/final_model_binary.joblib"):
            pytest.skip("binary model artifact not built yet")
        pipe = ADNIInferencePipeline(mode="binary", model_dir=MODEL_DIR)
        result = pipe.predict_one(VALID_RECORD, explain=False)
        assert result.model_accuracy["point_estimate"] >= 0.80


class TestEdgeCases:
    def test_missing_single_biomarker(self, pipeline):
        rec = dict(VALID_RECORD); rec["GFAP_F"] = None
        result = pipeline.predict_one(rec, explain=False)
        assert result.diagnosis in ["Control", "MCI", "Dementia"]
        assert any("GFAP_F" in w for w in result.warnings)

    def test_negative_value_treated_as_sentinel(self, pipeline):
        rec = dict(VALID_RECORD); rec["NfL_F"] = -4.0
        result = pipeline.predict_one(rec, explain=False)
        assert any("negative" in w.lower() for w in result.warnings)

    def test_zero_value_passes_through(self, pipeline):
        rec = dict(VALID_RECORD); rec["AB42_AB40_F"] = 0.0
        result = pipeline.predict_one(rec, explain=False)
        assert result.diagnosis in ["Control", "MCI", "Dementia"]

    def test_nan_value_handled(self, pipeline):
        rec = dict(VALID_RECORD); rec["pT217_AB42_F"] = float("nan")
        result = pipeline.predict_one(rec, explain=False)
        assert result.diagnosis in ["Control", "MCI", "Dementia"]

    def test_extremely_large_value_flagged_not_rejected(self, pipeline):
        rec = dict(VALID_RECORD); rec["GFAP_F"] = 1e7
        result = pipeline.predict_one(rec, explain=False)
        assert any("extremely large" in w for w in result.warnings)

    def test_invalid_datatype_raises_validation_error(self, pipeline):
        rec = dict(VALID_RECORD); rec["pT217_F"] = "not_a_number"
        with pytest.raises(InputValidationError):
            pipeline.predict_one(rec, explain=False)

    def test_all_biomarkers_missing_raises(self, pipeline):
        rec = {k: None for k in VALID_RECORD}
        with pytest.raises(InputValidationError):
            pipeline.predict_one(rec, explain=False)

    def test_empty_dict_raises(self, pipeline):
        with pytest.raises(InputValidationError):
            pipeline.predict_one({}, explain=False)

    def test_unknown_threshold_mode_raises(self, pipeline):
        with pytest.raises(InputValidationError):
            pipeline.predict_one(VALID_RECORD, threshold_mode="not_a_real_mode")

    def test_pT217_missing_adds_confidence_warning(self, pipeline):
        rec = dict(VALID_RECORD); rec["pT217_F"] = None
        result = pipeline.predict_one(rec, explain=False)
        assert any("most influential feature" in w for w in result.warnings)

    def test_batch_prediction(self, pipeline):
        results = pipeline.predict_batch([VALID_RECORD, VALID_RECORD], explain=False)
        assert len(results) == 2

    def test_unknown_mode_raises_at_construction(self):
        with pytest.raises(ValueError):
            ADNIInferencePipeline(mode="not_a_real_mode", model_dir=MODEL_DIR)


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])

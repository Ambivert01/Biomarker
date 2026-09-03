"""
inference.py
=============
Production inference pipeline. Supports BOTH models built for this project:

  mode="binary"      -> Control vs Dementia only (Random Forest, 87.4% test accuracy)
                         RECOMMENDED — this is what was selected after the
                         3-class model could not reliably clear 80% accuracy.
  mode="three_class"  -> Control vs MCI vs Dementia (Extra Trees, 61.2% test accuracy)
                         Kept available for reference / research use.

Implements the flow requested in the project brief:
    Input -> Validation -> Feature Engineering -> Scaling -> Prediction ->
    Probability -> Confidence -> SHAP Explanation -> Structured JSON Response

Every response includes the model's own validated accuracy (point estimate +
95% confidence interval, measured once on a held-out test set never used
for training or tuning) so the number travels with every prediction, not
just in a separate report.

This is the single entry point a REST API, batch job, or Streamlit app
should call. It never re-fits anything — it only loads persisted artifacts.
"""
from __future__ import annotations
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Pickled model artifacts reference our custom transformer classes by their
# module path (e.g. `preprocessing.pipeline.FeatureSetSelector`), so `src/`
# must be importable regardless of the caller's cwd.
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd
import joblib
import shap

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Per-mode configuration: which files to load, and the class label mapping.
# ---------------------------------------------------------------------------
MODE_CONFIG = {
    "binary": dict(
        model_filename="final_model_binary.joblib",
        uncalibrated_filename="final_model_binary_uncalibrated.joblib",
        metadata_filename="final_model_binary_metadata.json",
        class_order=[1, 3],
        class_names={1: "Control", 3: "Dementia"},
    ),
    "three_class": dict(
        model_filename="final_model_biomarker_only.joblib",
        uncalibrated_filename="final_model_biomarker_only_uncalibrated.joblib",
        metadata_filename="final_model_biomarker_only_metadata.json",
        class_order=[1, 2, 3],
        class_names={1: "Control", 2: "MCI", 3: "Dementia"},
    ),
}

# The exact raw input columns the model needs. Anything else in a request is
# ignored; anything missing from this list is imputed downstream (harmonized
# from whichever assay platform is present, or median-imputed) — the model
# is explicitly designed to tolerate partial biomarker panels.
REQUIRED_RAW_COLUMNS = [
    "pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F",
    "NfL_F", "GFAP_F", "NfL_Q", "GFAP_Q",
]

# Dementia decision-threshold operating points, per mode. For "binary" these
# shift the Control/Dementia cutoff away from the default 0.5; for
# "three_class" they follow the same "predict Dementia if P(Dementia)>=t,
# else argmax(Control,MCI)" rule documented in the report.
THRESHOLD_MODES = {
    "binary": {"standard_argmax": 0.5, "balanced": 0.5, "high_sensitivity": 0.35},
    "three_class": {"standard_argmax": None, "balanced": 0.35, "high_sensitivity": 0.225},
}


class InputValidationError(Exception):
    pass


@dataclass
class PredictionResult:
    diagnosis: str
    diagnosis_code: int
    probabilities: dict
    confidence: float
    diagnostic_scope: str
    threshold_mode: str
    decision_rule_note: str
    model_accuracy: dict
    top_shap_contributors: Optional[list]
    warnings: list
    model_version: str

    def to_json(self, indent=2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)


class ADNIInferencePipeline:
    """
    Loads a persisted model once, then serves predictions.

    Example
    -------
    >>> pipe = ADNIInferencePipeline(mode="binary")   # Control vs Dementia, 87.4% test accuracy
    >>> result = pipe.predict_one({"pT217_F": 0.85, "AB42_F": 26.5, "AB40_F": 355.0,
    ...                             "AB42_AB40_F": 0.075, "pT217_AB42_F": 0.032,
    ...                             "NfL_F": 45.0, "GFAP_F": 110.0, "NfL_Q": None, "GFAP_Q": None})
    >>> print(result.to_json())
    """

    def __init__(self, mode: str = "binary", model_dir: str = None):
        if mode not in MODE_CONFIG:
            raise ValueError(f"mode must be one of {list(MODE_CONFIG)}, got '{mode}'")
        self.mode = mode
        cfg = MODE_CONFIG[mode]
        self.class_order = cfg["class_order"]
        self.class_names = cfg["class_names"]

        if model_dir is None:
            # Resolve relative to this file's location (project_root/src/api/inference.py
            # -> project_root/artifacts/models), so it works regardless of the caller's cwd.
            model_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "models"
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / cfg["model_filename"])
        self.raw_pipe = joblib.load(model_dir / cfg["uncalibrated_filename"])
        with open(model_dir / cfg["metadata_filename"]) as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata["features"]
        self._shap_explainer = None

        acc = self.metadata["test_metrics"]["accuracy"]
        ci = self.metadata["bootstrap_ci_95"]["accuracy"]
        logger.info(f"Loaded model '{self.metadata['model_name']}' (mode={mode}, role: {self.metadata['role']}, "
                     f"calibration: {self.metadata['calibration_method']}) "
                     f"-- validated test accuracy {acc:.1%} (95% CI [{ci[1]:.1%}, {ci[2]:.1%}], n={self.metadata['n_test']})")

    def _model_accuracy_block(self) -> dict:
        acc = self.metadata["test_metrics"]["accuracy"]
        acc_mean, acc_lo, acc_hi = self.metadata["bootstrap_ci_95"]["accuracy"]
        return {
            "point_estimate": round(acc, 4),
            "ci_95_low": round(acc_lo, 4),
            "ci_95_high": round(acc_hi, 4),
            "n_test_patients": self.metadata["n_test"],
            "note": ("Validated ONCE on a held-out test set never used for training or tuning. "
                     "This is the model's overall accuracy across many patients, not a per-prediction guarantee "
                     "for this specific patient."),
        }

    # ------------------------------------------------------------------
    # 1. VALIDATION
    # ------------------------------------------------------------------
    def _validate_and_clean(self, record: dict) -> tuple[pd.DataFrame, list]:
        warnings_list = []
        clean = {}
        for col in REQUIRED_RAW_COLUMNS:
            val = record.get(col, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                warnings_list.append(f"'{col}' missing from input — will be imputed/harmonized from other available data.")
                clean[col] = np.nan
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                raise InputValidationError(f"'{col}' must be numeric, got {val!r}")
            if val != val:  # NaN check
                clean[col] = np.nan
                continue
            if val < 0:
                warnings_list.append(f"'{col}'={val} is negative (physically impossible concentration) "
                                      f"— treated as a below-limit-of-quantification sentinel and set to missing.")
                clean[col] = np.nan
                continue
            if val > 1e5:
                warnings_list.append(f"'{col}'={val} is an extremely large value; passed through but flagged for review.")
            clean[col] = val

        n_present = sum(1 for v in clean.values() if v == v)  # not NaN
        if n_present == 0:
            raise InputValidationError("All biomarker fields are missing/invalid — cannot make a prediction from no data.")
        if clean.get("pT217_F") != clean.get("pT217_F"):  # NaN check (pT217_F is the single strongest predictor)
            warnings_list.append("pT217_F (the model's most influential feature) is missing — "
                                  "prediction confidence is likely to be substantially reduced.")

        df = pd.DataFrame([clean])
        for extra_col in ["MMSE_FINAL_SCORE"]:
            df[extra_col] = np.nan
        return df, warnings_list

    # ------------------------------------------------------------------
    # 2-5. FEATURE ENGINEERING -> SCALING -> PREDICTION -> PROBABILITY (all inside the fitted pipeline)
    # ------------------------------------------------------------------
    def _predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df)

    # ------------------------------------------------------------------
    # 6. SHAP EXPLANATION
    # ------------------------------------------------------------------
    def _get_shap_explainer(self):
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.raw_pipe.named_steps["clf"])
        return self._shap_explainer

    def _explain(self, df: pd.DataFrame, predicted_class_idx: int, top_k: int = 5) -> list:
        prep = self.raw_pipe.named_steps["prep"]
        X_transformed = prep.transform(df)
        explainer = self._get_shap_explainer()
        shap_values = explainer(pd.DataFrame(X_transformed, columns=self.feature_names))
        contribs = shap_values.values[0, :, predicted_class_idx]
        order = np.argsort(np.abs(contribs))[::-1][:top_k]
        return [
            {"feature": self.feature_names[i], "shap_value": float(contribs[i]),
             "direction": "increases" if contribs[i] > 0 else "decreases"}
            for i in order
        ]

    # ------------------------------------------------------------------
    # 7. STRUCTURED RESPONSE
    # ------------------------------------------------------------------
    def predict_one(self, record: dict, threshold_mode: str = "balanced", explain: bool = True) -> PredictionResult:
        mode_thresholds = THRESHOLD_MODES[self.mode]
        if threshold_mode not in mode_thresholds:
            raise InputValidationError(f"threshold_mode must be one of {list(mode_thresholds)}, got '{threshold_mode}'")

        df, warnings_list = self._validate_and_clean(record)
        proba = self._predict_proba(df)[0]
        t = mode_thresholds[threshold_mode]
        dementia_idx = self.class_order.index(3)

        if self.mode == "three_class":
            if t is None:
                pred_idx = int(np.argmax(proba))
                pred_code = self.class_order[pred_idx]
                rule_note = "Standard argmax over calibrated class probabilities."
            elif proba[dementia_idx] >= t:
                pred_code = 3
                rule_note = f"P(Dementia)={proba[dementia_idx]:.3f} >= threshold {t} -> predicted Dementia (sensitivity-adjusted rule)."
            else:
                remaining_idx = [i for i in range(len(self.class_order)) if i != dementia_idx]
                best = remaining_idx[int(np.argmax(proba[remaining_idx]))]
                pred_code = self.class_order[best]
                rule_note = (f"P(Dementia)={proba[dementia_idx]:.3f} < threshold {t} -> predicted "
                             f"{self.class_names[pred_code]} via argmax(Control, MCI).")
        else:  # binary
            if proba[dementia_idx] >= t:
                pred_code = 3
                rule_note = f"P(Dementia)={proba[dementia_idx]:.3f} >= threshold {t} -> predicted Dementia."
            else:
                pred_code = 1
                rule_note = f"P(Dementia)={proba[dementia_idx]:.3f} < threshold {t} -> predicted Control."

        pred_class_idx = self.class_order.index(pred_code)
        confidence = float(proba[pred_class_idx])

        top_contribs = None
        if explain:
            try:
                top_contribs = self._explain(df, pred_class_idx)
            except Exception as e:
                warnings_list.append(f"SHAP explanation unavailable: {e}")

        return PredictionResult(
            diagnosis=self.class_names[pred_code],
            diagnosis_code=pred_code,
            probabilities={self.class_names[c]: float(p) for c, p in zip(self.class_order, proba)},
            confidence=confidence,
            diagnostic_scope=("Control vs Dementia only (MCI not modeled in this mode)"
                               if self.mode == "binary" else "Control vs MCI vs Dementia"),
            threshold_mode=threshold_mode,
            decision_rule_note=rule_note,
            model_accuracy=self._model_accuracy_block(),
            top_shap_contributors=top_contribs,
            warnings=warnings_list,
            model_version=f"{self.metadata['model_name']}_{self.mode}_{self.metadata['calibration_method']}_v1",
        )

    def predict_batch(self, records: list[dict], threshold_mode: str = "balanced", explain: bool = False) -> list[PredictionResult]:
        return [self.predict_one(r, threshold_mode=threshold_mode, explain=explain) for r in records]


if __name__ == "__main__":
    print("=" * 70)
    print("BINARY model (Control vs Dementia) -- recommended, 87.4% test accuracy")
    print("=" * 70)
    pipe = ADNIInferencePipeline(mode="binary")
    example = {
        "pT217_F": 0.85, "AB42_F": 26.5, "AB40_F": 355.0,
        "AB42_AB40_F": 0.075, "pT217_AB42_F": 0.032,
        "NfL_F": 45.0, "GFAP_F": 110.0, "NfL_Q": None, "GFAP_Q": None,
    }
    result = pipe.predict_one(example, threshold_mode="balanced", explain=True)
    print(result.to_json())

    print("\n" + "=" * 70)
    print("THREE-CLASS model (Control vs MCI vs Dementia) -- reference, 61.2% test accuracy")
    print("=" * 70)
    pipe3 = ADNIInferencePipeline(mode="three_class")
    result3 = pipe3.predict_one(example, threshold_mode="balanced", explain=True)
    print(result3.to_json())

"""
08_train_biomarker_only.py
=============================
Runs the SAME 9-model Optuna bake-off as 06_train_all_models.py, but on the
BIOMARKER-ONLY feature set ("core_plus_engineered": 7 core biomarkers + 12
engineered ratio/log/rank features, NO MMSE, NO phase).

Why a second bake-off instead of reusing 06's results:
This session's audit (see reports/DATA_AUDIT_REPORT.md and the "MMSE
circularity" experiment run in this session) found that:
  (a) MMSE missingness is highly informative of diagnosis for
      non-biological (clinical-documentation) reasons, and
  (b) MMSE alone (f1_macro=0.57) is nearly as predictive as the ENTIRE
      biomarker panel (f1_macro=0.55) — because MMSE is one of the
      instruments ADNI's own diagnostic algorithm uses to assign the MCI /
      Dementia label in the first place, making an "MMSE-augmented"
      biomarker model uncomfortably close to circular.
The project's own Primary Objective is a classifier that predicts
diagnosis "purely from plasma biomarker concentrations" — so THIS
biomarker-only model, not the earlier MMSE-augmented one, is the correct
candidate for that objective and becomes the production/default model.
The earlier MMSE-augmented LightGBM model (06/07) is retained and reported
as a clearly-labeled secondary reference/upper-bound, not deployed.

Everything else (splits, CV, Optuna budget, scoring) is identical to 06 for
a fair comparison.
"""
import sys, os, json, time, warnings
sys.path.append('../src')
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline

from data_loader import load_and_validate
from preprocessing.pipeline import build_full_preprocessing_pipeline, get_feature_set
from models.model_zoo import build_model, build_model_with_params, MODEL_NAMES
from evaluation.metrics import compute_full_metrics

RANDOM_SEED = 42
N_TRIALS = 40
TIMEOUT_PER_MODEL = 240
CV_FOLDS = 5
FEATURE_SET_NAME = "core_plus_engineered"     # <-- biomarker-only, the key change from 06
IMPUTE_STRATEGY = "median"
SCALE_STRATEGY = "robust"
RESULTS_CSV = "../reports/model_bakeoff_cv_results_biomarker_only.csv"
PARAMS_DIR = "../artifacts/optuna_studies_biomarker_only"

os.makedirs(PARAMS_DIR, exist_ok=True)

print("Loading data...")
df, _ = load_and_validate("../config/config.yaml")
X_all = df.drop(columns=["DIAGNOSIS"])
y_all = df["DIAGNOSIS"]

# SAME stratified holdout split (same random_state) as 06/07, so the dev/test
# partition is identical across both feature-set experiments -> a fair,
# apples-to-apples comparison and a single held-out test set used by both.
X_dev, X_test, y_dev, y_test = train_test_split(
    X_all, y_all, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED)
print(f"Dev pool: {X_dev.shape[0]} | Held-out test: {X_test.shape[0]} (untouched)")

feats = get_feature_set(FEATURE_SET_NAME)
print(f"Feature set '{FEATURE_SET_NAME}': {len(feats)} features -> {feats}")
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


def make_pipeline_for_trial(model_name, trial):
    prep = build_full_preprocessing_pipeline(
        numeric_features=feats, imputation_strategy=IMPUTE_STRATEGY, scaling_strategy=SCALE_STRATEGY)
    clf = build_model(model_name, trial, random_state=RANDOM_SEED)
    return Pipeline([("prep", prep), ("clf", clf)])


def objective_factory(model_name):
    def objective(trial):
        pipe = make_pipeline_for_trial(model_name, trial)
        try:
            scores = cross_val_score(pipe, X_dev, y_dev, cv=cv, scoring="f1_macro", n_jobs=1, error_score="raise")
        except Exception as e:
            raise optuna.TrialPruned(str(e))
        return scores.mean()
    return objective


bake_off_results = []

for model_name in MODEL_NAMES:
    print(f"\n{'='*70}\nTuning: {model_name}\n{'='*70}")
    t0 = time.time()
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_factory(model_name), n_trials=N_TRIALS, timeout=TIMEOUT_PER_MODEL, show_progress_bar=False)
    elapsed = time.time() - t0

    best_params = study.best_params

    class DummyTrial:
        def __init__(self, params): self.params = params
        def suggest_float(self, name, *a, **k): return self.params[name]
        def suggest_int(self, name, *a, **k): return self.params[name]
        def suggest_categorical(self, name, *a, **k): return self.params[name]

    best_pipe = make_pipeline_for_trial(model_name, DummyTrial(best_params))
    dev_preds = cross_val_predict(best_pipe, X_dev, y_dev, cv=cv, n_jobs=1)
    dev_metrics = compute_full_metrics(y_dev, dev_preds)

    result = {
        "model": model_name,
        "cv_macro_f1_mean": study.best_value,
        "cv_macro_f1_std": float(np.std([t.value for t in study.trials if t.value is not None])),
        "cv_balanced_accuracy": dev_metrics["balanced_accuracy"],
        "cv_dementia_recall": dev_metrics["recall_Dementia"],
        "cv_dementia_precision": dev_metrics["precision_Dementia"],
        "cv_control_recall": dev_metrics["recall_Control"],
        "cv_mci_recall": dev_metrics["recall_MCI"],
        "n_trials_completed": len(study.trials),
        "tuning_seconds": round(elapsed, 1),
    }
    bake_off_results.append(result)
    print(f"  Best CV macro-F1: {study.best_value:.4f} | Dementia recall: {dev_metrics['recall_Dementia']:.4f} "
          f"| ({len(study.trials)} trials, {elapsed:.1f}s)")

    with open(f"{PARAMS_DIR}/{model_name}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # incremental save after each model, so progress survives any interruption
    pd.DataFrame(bake_off_results).sort_values("cv_macro_f1_mean", ascending=False).to_csv(RESULTS_CSV, index=False)

bake_off_df = pd.DataFrame(bake_off_results).sort_values("cv_macro_f1_mean", ascending=False)
print("\n\n" + "="*90)
print(f"MODEL BAKE-OFF (BIOMARKER-ONLY, feature_set={FEATURE_SET_NAME}) — 5-fold CV on dev pool")
print("="*90)
print(bake_off_df.to_string(index=False))
print(f"\nSaved: {RESULTS_CSV}")

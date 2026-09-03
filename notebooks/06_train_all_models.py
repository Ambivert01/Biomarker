"""
train_pipeline.py
===================
Implements the training strategy required by the project brief:

    Stratified Holdout -> Train/Dev pool -> Stratified K-Fold CV -> Optuna HPO
    -> [model selection happens here, using CV metrics ONLY] -> Final Test

The held-out test set (15%, stratified) is NEVER touched until the very end,
and then only to evaluate the single already-selected, already-tuned final
model. All 9 models are compared using 5-fold stratified cross-validation
on the 85% dev pool — this is what determines the bake-off ranking.

Run: python3 06_train_all_models.py
"""
import sys, os, json, time, warnings, pickle
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
from evaluation.metrics import compute_full_metrics, format_metrics_report

RANDOM_SEED = 42
N_TRIALS = 40
CV_FOLDS = 5
FEATURE_SET_NAME = "core_plus_engineered_plus_mmse"   # decided in notebooks/05_feature_ablation.py
IMPUTE_STRATEGY = "median"                              # decided in notebooks/03_imputation_comparison.py
SCALE_STRATEGY = "robust"

os.makedirs("../artifacts/models", exist_ok=True)
os.makedirs("../artifacts/optuna_studies", exist_ok=True)

print("Loading data...")
df, _ = load_and_validate("../config/config.yaml")
X_all = df.drop(columns=["DIAGNOSIS"])
y_all = df["DIAGNOSIS"]

# ---------------------------------------------------------------------------
# STRATIFIED HOLDOUT — test set carved out ONCE, right here, never seen again
# until the final model evaluation cell at the bottom of this script.
# ---------------------------------------------------------------------------
X_dev, X_test, y_dev, y_test = train_test_split(
    X_all, y_all, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED)
print(f"Dev pool: {X_dev.shape[0]} patients | Held-out test: {X_test.shape[0]} patients (untouched until final eval)")
print(f"Dev class distribution: {y_dev.value_counts().to_dict()}")
print(f"Test class distribution: {y_test.value_counts().to_dict()}")

feats = get_feature_set(FEATURE_SET_NAME)
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
best_params_by_model = {}

for model_name in MODEL_NAMES:
    print(f"\n{'='*70}\nTuning: {model_name}\n{'='*70}")
    t0 = time.time()
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_factory(model_name), n_trials=N_TRIALS, timeout=300, show_progress_bar=False)
    elapsed = time.time() - t0

    best_params = study.best_params
    best_params_by_model[model_name] = best_params

    # Rebuild the best pipeline and get a fuller CV metric picture (macro-F1,
    # balanced accuracy, dementia recall) via cross_val_predict -- still
    # entirely within the dev pool, no test-set contact.
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
        "cv_macro_f1_std": np.std([t.value for t in study.trials if t.value is not None]),
        "cv_balanced_accuracy": dev_metrics["balanced_accuracy"],
        "cv_dementia_recall": dev_metrics["recall_Dementia"],
        "cv_dementia_precision": dev_metrics["precision_Dementia"],
        "cv_control_recall": dev_metrics["recall_Control"],
        "cv_mci_recall": dev_metrics["recall_MCI"],
        "n_trials_completed": len(study.trials),
        "tuning_seconds": elapsed,
    }
    bake_off_results.append(result)
    print(f"  Best CV macro-F1: {study.best_value:.4f} | Dementia recall: {dev_metrics['recall_Dementia']:.4f} "
          f"| ({len(study.trials)} trials, {elapsed:.1f}s)")

    with open(f"../artifacts/optuna_studies/{model_name}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

bake_off_df = pd.DataFrame(bake_off_results).sort_values("cv_macro_f1_mean", ascending=False)
bake_off_df.to_csv("../reports/model_bakeoff_cv_results.csv", index=False)
print("\n\n" + "="*90)
print("MODEL BAKE-OFF — 5-FOLD STRATIFIED CV ON DEV POOL (test set NOT used)")
print("="*90)
print(bake_off_df.to_string(index=False))

with open("../artifacts/models/best_params_by_model.json", "w") as f:
    json.dump(best_params_by_model, f, indent=2)

print("\nSaved: reports/model_bakeoff_cv_results.csv, artifacts/models/best_params_by_model.json")
print("\nDONE — model selection happens in the next script (07_select_and_finalize.py),")
print("using this CV table plus the clinical dementia-recall preference. Test set still untouched.")

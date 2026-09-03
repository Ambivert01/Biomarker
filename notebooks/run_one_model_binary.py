"""
run_one_model_binary.py
=========================
Optuna bake-off for the BINARY reframing: Control vs. Dementia only (MCI
patients excluded from training and evaluation entirely). Same feature set,
same leakage-safe pipeline, same rigor as the primary 3-class model — this
is a real bake-off, not a reuse of the 3-class model's hyperparameters.

Usage: python3 run_one_model_binary.py <model_name> [n_trials] [timeout_seconds]
"""
import sys, os, json, time, warnings
sys.path.append('../src')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline

from data_loader import load_and_validate
from preprocessing.pipeline import build_full_preprocessing_pipeline, get_feature_set
from models.model_zoo import build_model
from evaluation.metrics import compute_full_metrics

RANDOM_SEED = 42
FEATURE_SET_NAME = "core_plus_engineered"
IMPUTE_STRATEGY = "median"
SCALE_STRATEGY = "robust"
CV_FOLDS = 5
RESULTS_CSV = "../reports/model_bakeoff_cv_results_binary.csv"
PARAMS_DIR = "../artifacts/optuna_studies_binary"

model_name = sys.argv[1]
n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 40
timeout_s = int(sys.argv[3]) if len(sys.argv) > 3 else 140

os.makedirs(PARAMS_DIR, exist_ok=True)

df, _ = load_and_validate("../config/config.yaml")
binary_df = df[df["DIAGNOSIS"].isin([1, 3])].copy()   # Control=1, Dementia=3 ; MCI=2 dropped
X_all = binary_df.drop(columns=["DIAGNOSIS"])
y_all = binary_df["DIAGNOSIS"]
X_dev, X_test, y_dev, y_test = train_test_split(
    X_all, y_all, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED)

feats = get_feature_set(FEATURE_SET_NAME)
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


def make_pipeline_for_trial(trial):
    prep = build_full_preprocessing_pipeline(
        numeric_features=feats, imputation_strategy=IMPUTE_STRATEGY, scaling_strategy=SCALE_STRATEGY)
    clf = build_model(model_name, trial, random_state=RANDOM_SEED, n_classes=2)
    return Pipeline([("prep", prep), ("clf", clf)])


def objective(trial):
    pipe = make_pipeline_for_trial(trial)
    scores = cross_val_score(pipe, X_dev, y_dev, cv=cv, scoring="accuracy", n_jobs=1, error_score="raise")
    return scores.mean()


print(f"Tuning {model_name} [BINARY Control-vs-Dementia]: n_trials={n_trials}, timeout={timeout_s}s")
print(f"Dev pool: {len(X_dev)} (Control={sum(y_dev==1)}, Dementia={sum(y_dev==3)}) | Test: {len(X_test)}")
t0 = time.time()
sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
elapsed = time.time() - t0

best_params = study.best_params


class DummyTrial:
    def __init__(self, params): self.params = params
    def suggest_float(self, name, *a, **k): return self.params[name]
    def suggest_int(self, name, *a, **k): return self.params[name]
    def suggest_categorical(self, name, *a, **k): return self.params[name]


best_pipe = make_pipeline_for_trial(DummyTrial(best_params))
dev_preds = cross_val_predict(best_pipe, X_dev, y_dev, cv=cv, n_jobs=1)
dev_metrics = compute_full_metrics(y_dev, dev_preds, class_order=[1, 3], class_names=["Control", "Dementia"])

result = {
    "model": model_name,
    "cv_accuracy_mean": study.best_value,
    "cv_accuracy_std": float(np.std([t.value for t in study.trials if t.value is not None])),
    "cv_macro_f1": dev_metrics["macro_f1"],
    "cv_dementia_recall": dev_metrics["recall_Dementia"],
    "cv_dementia_precision": dev_metrics["precision_Dementia"],
    "cv_control_recall": dev_metrics["recall_Control"],
    "n_trials_completed": len(study.trials),
    "tuning_seconds": round(elapsed, 1),
}
print(json.dumps(result, indent=2))

with open(f"{PARAMS_DIR}/{model_name}_best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

file_exists = os.path.exists(RESULTS_CSV)
existing = pd.read_csv(RESULTS_CSV) if file_exists else pd.DataFrame()
existing = existing[existing["model"] != model_name] if len(existing) else existing
existing = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)
existing.to_csv(RESULTS_CSV, index=False)
print(f"\nAppended to {RESULTS_CSV}. Total models completed so far: {len(existing)}")

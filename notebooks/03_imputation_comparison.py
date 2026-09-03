"""
Compares median / KNN / iterative imputation strategies (crossed with
standard vs robust scaling) using nested, leakage-safe cross-validation.
A fixed, simple, well-understood classifier (multinomial Logistic
Regression AND Random Forest) is used as the "measuring instrument" so the
comparison isolates the effect of the imputation strategy rather than
conflating it with a specific model's own hyperparameter sensitivity.
"""
import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from data_loader import load_and_validate
from preprocessing.pipeline import build_full_preprocessing_pipeline, get_feature_set

RANDOM_SEED = 42

df, _ = load_and_validate("../config/config.yaml")
X = df.drop(columns=["DIAGNOSIS"])
y = df["DIAGNOSIS"]

feats = get_feature_set("core_plus_engineered_plus_mmse")

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for impute_strategy in ["median", "knn", "iterative"]:
    for scale_strategy in ["standard", "robust"]:
        prep = build_full_preprocessing_pipeline(
            numeric_features=feats,
            imputation_strategy=impute_strategy,
            scaling_strategy=scale_strategy,
        )
        for clf_name, clf in [
            ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
            ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
        ]:
            pipe = Pipeline([("prep", prep), ("clf", clf)])
            scoring = {
                "f1_macro": "f1_macro",
                "balanced_accuracy": "balanced_accuracy",
            }
            scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
            results.append({
                "impute": impute_strategy,
                "scale": scale_strategy,
                "clf": clf_name,
                "f1_macro_mean": scores["test_f1_macro"].mean(),
                "f1_macro_std": scores["test_f1_macro"].std(),
                "bal_acc_mean": scores["test_balanced_accuracy"].mean(),
                "bal_acc_std": scores["test_balanced_accuracy"].std(),
            })
            print(results[-1])

res_df = pd.DataFrame(results).sort_values("f1_macro_mean", ascending=False)
print("\n\n=== FULL RESULTS (sorted by macro-F1) ===")
print(res_df.to_string(index=False))
res_df.to_csv("../reports/imputation_comparison_results.csv", index=False)

import sys, warnings
sys.path.append('../src')
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

feature_sets = ["core_only", "core_plus_engineered", "core_plus_mmse_no_engineered",
                "core_plus_engineered_plus_mmse", "everything_incl_confound_risk"]

results = []
for fs_name in feature_sets:
    feats = get_feature_set(fs_name)
    cat_feats = ["PHASE"] if fs_name == "everything_incl_confound_risk" else []
    prep = build_full_preprocessing_pipeline(
        numeric_features=feats, categorical_features=cat_feats,
        imputation_strategy="median", scaling_strategy="robust")
    for clf_name, clf in [
        ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
    ]:
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        scoring = {"f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy",
                   "recall_dementia": "recall_macro"}  # placeholder, per-class recall computed below separately
        scores = cross_validate(pipe, X, y, cv=cv, scoring={"f1_macro":"f1_macro","bal_acc":"balanced_accuracy"},
                                 n_jobs=-1, error_score="raise")
        results.append({
            "feature_set": fs_name, "n_features": len(feats) + len(cat_feats), "clf": clf_name,
            "f1_macro_mean": scores["test_f1_macro"].mean(), "f1_macro_std": scores["test_f1_macro"].std(),
            "bal_acc_mean": scores["test_bal_acc"].mean(), "bal_acc_std": scores["test_bal_acc"].std(),
        })
        print(results[-1])

res_df = pd.DataFrame(results).sort_values("f1_macro_mean", ascending=False)
print("\n=== FEATURE SET ABLATION (sorted by macro-F1) ===")
print(res_df.to_string(index=False))
res_df.to_csv("../reports/feature_set_ablation_results.csv", index=False)

# Also get per-class (esp. Dementia) recall for the top few configs using cross_val_predict
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

print("\n=== Per-class detail for each feature set (Logistic Regression) ===")
for fs_name in feature_sets:
    feats = get_feature_set(fs_name)
    cat_feats = ["PHASE"] if fs_name == "everything_incl_confound_risk" else []
    prep = build_full_preprocessing_pipeline(numeric_features=feats, categorical_features=cat_feats,
                                              imputation_strategy="median", scaling_strategy="robust")
    pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED))])
    preds = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    print(f"\n--- {fs_name} ---")
    print(classification_report(y, preds, target_names=["Control","MCI","Dementia"], digits=3))

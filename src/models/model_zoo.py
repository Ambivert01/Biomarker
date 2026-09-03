"""
model_zoo.py
=============
One place defining every candidate model and its Optuna search space, so
training/train_pipeline.py stays a thin orchestration loop instead of a
9-way copy-pasted mess. Each `suggest_*` function takes an optuna.Trial and
returns a fitted-but-not-fit sklearn/xgboost/lightgbm/catboost estimator
ready to drop into the shared preprocessing Pipeline.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

MODEL_NAMES = [
    "logistic_regression", "random_forest", "extra_trees", "xgboost", "lightgbm",
    "catboost", "hist_gradient_boosting", "svm", "mlp",
]


def suggest_logistic_regression(trial, random_state):
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    penalty = trial.suggest_categorical("penalty", ["l2", "l1"])
    solver = "saga" if penalty == "l1" else "lbfgs"
    return LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=3000,
                               class_weight="balanced", random_state=random_state)


def suggest_random_forest(trial, random_state):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 16),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, None]),
        class_weight=trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        random_state=random_state, n_jobs=1,
    )


def suggest_extra_trees(trial, random_state):
    return ExtraTreesClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 16),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, None]),
        class_weight=trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        random_state=random_state, n_jobs=1,
    )


class XGBWrapper(BaseEstimator, ClassifierMixin):
    """
    XGBoost's sklearn API requires contiguous 0-indexed class labels, but the
    rest of this project deliberately keeps DIAGNOSIS in its native ADNI
    encoding (1=Control, 2=MCI, 3=Dementia -- and for the binary Control-vs-
    Dementia task, just {1, 3}, which is NOT contiguous) for traceability
    back to the source data. This wrapper maps whatever label set is present
    to 0..k-1 via np.unique's inverse index (not a naive min-subtraction,
    which breaks on non-contiguous labels like {1, 3}) and maps predictions
    back on the way out. predict_proba's column order follows self.classes_
    (ascending), matching sklearn convention.
    """

    def __init__(self, xgb_kwargs=None):
        self.xgb_kwargs = xgb_kwargs

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        _, y_encoded = np.unique(y, return_inverse=True)
        self._model_ = XGBClassifier(**(self.xgb_kwargs or {}))
        self._model_.fit(X, y_encoded)
        return self

    def predict(self, X):
        encoded_pred = self._model_.predict(X)
        return self.classes_[encoded_pred.astype(int)]

    def predict_proba(self, X):
        return self._model_.predict_proba(X)


def suggest_xgboost(trial, random_state, n_classes: int = 3):
    xgb_objective_kwargs = (
        dict(objective="binary:logistic", eval_metric="logloss")
        if n_classes == 2 else
        dict(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")
    )
    kwargs = dict(
        n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 15),
        gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        random_state=random_state, n_jobs=1, verbosity=0,
        **xgb_objective_kwargs,
    )
    return XGBWrapper(xgb_kwargs=kwargs)


def suggest_lightgbm(trial, random_state):
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 12),
        num_leaves=trial.suggest_int("num_leaves", 7, 128),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 3, 40),
        class_weight="balanced",
        random_state=random_state, n_jobs=1, verbosity=-1,
    )


def suggest_catboost(trial, random_state):
    # NOTE: depth is by far the dominant cost driver on this sandbox's single
    # CPU core (depth=10/iterations=400 measured at ~49s for ONE fit, making
    # 5-fold CV infeasible within the compute budget). Search space is
    # narrowed accordingly; in a production environment with more cores this
    # would be widened back toward CatBoost's usual depth<=10 range.
    return CatBoostClassifier(
        iterations=trial.suggest_int("iterations", 50, 250, step=25),
        depth=trial.suggest_int("depth", 2, 6),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 20.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 3.0),
        random_strength=trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
        auto_class_weights="Balanced", thread_count=1,
        random_state=random_state, verbose=False, allow_writing_files=False,
    )


def suggest_hist_gradient_boosting(trial, random_state):
    return HistGradientBoostingClassifier(
        max_iter=trial.suggest_int("max_iter", 50, 400, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 16),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 40),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 7, 63),
        class_weight="balanced",
        random_state=random_state,
    )


def suggest_svm(trial, random_state):
    kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "linear"])
    params = dict(
        C=trial.suggest_float("C", 1e-2, 1e2, log=True),
        kernel=kernel,
        gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
        probability=True, class_weight="balanced", random_state=random_state,
    )
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 4)
    return SVC(**params)


def suggest_mlp(trial, random_state):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_size = trial.suggest_categorical("layer_size", [16, 32, 64, 128])
    hidden = tuple([layer_size] * n_layers)
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        alpha=trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
        early_stopping=True, n_iter_no_change=15, max_iter=1000,
        random_state=random_state,
    )


SUGGEST_FUNCS = {
    "logistic_regression": suggest_logistic_regression,
    "random_forest": suggest_random_forest,
    "extra_trees": suggest_extra_trees,
    "xgboost": suggest_xgboost,
    "lightgbm": suggest_lightgbm,
    "catboost": suggest_catboost,
    "hist_gradient_boosting": suggest_hist_gradient_boosting,
    "svm": suggest_svm,
    "mlp": suggest_mlp,
}


def build_model(name: str, trial, random_state: int = 42, n_classes: int = 3):
    if name not in SUGGEST_FUNCS:
        raise ValueError(f"Unknown model '{name}'. Options: {list(SUGGEST_FUNCS)}")
    if name == "xgboost":
        return suggest_xgboost(trial, random_state, n_classes=n_classes)
    return SUGGEST_FUNCS[name](trial, random_state)


def build_model_with_params(name: str, params: dict, random_state: int = 42, n_classes: int = 3):
    """Reconstruct a model from a saved best-params dict (used after HPO)."""
    ctor_map = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
        "catboost": CatBoostClassifier,
        "hist_gradient_boosting": HistGradientBoostingClassifier,
        "svm": SVC,
        "mlp": MLPClassifier,
    }
    params = dict(params)  # copy
    extra = {}
    if name == "xgboost":
        xgb_objective_kwargs = (
            dict(objective="binary:logistic", eval_metric="logloss")
            if n_classes == 2 else
            dict(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")
        )
        extra = dict(random_state=random_state, n_jobs=1, verbosity=0, **xgb_objective_kwargs)
        params.update(extra)
        return XGBWrapper(xgb_kwargs=params)
    if name == "logistic_regression":
        extra = dict(max_iter=3000, class_weight="balanced", random_state=random_state)
        if params.get("penalty") == "l1":
            params["solver"] = "saga"
        else:
            params["solver"] = "lbfgs"
    elif name in ("random_forest", "extra_trees"):
        extra = dict(random_state=random_state, n_jobs=1)
    elif name == "lightgbm":
        extra = dict(class_weight="balanced", random_state=random_state, n_jobs=1, verbosity=-1)
    elif name == "catboost":
        extra = dict(auto_class_weights="Balanced", thread_count=1, random_state=random_state, verbose=False, allow_writing_files=False)
    elif name == "hist_gradient_boosting":
        extra = dict(class_weight="balanced", random_state=random_state)
    elif name == "svm":
        extra = dict(probability=True, class_weight="balanced", random_state=random_state)
    elif name == "mlp":
        n_layers = params.pop("n_layers")
        layer_size = params.pop("layer_size")
        params["hidden_layer_sizes"] = tuple([layer_size] * n_layers)
        extra = dict(early_stopping=True, n_iter_no_change=15, max_iter=1000, random_state=random_state)

    params.update(extra)
    return ctor_map[name](**params)

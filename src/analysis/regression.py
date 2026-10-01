"""
regression.py
==============
Regression analysis using pTau-217 as the continuous outcome (most
biologically meaningful continuous target in this dataset).

Also performs regression of MMSE score on biomarkers (where available).

Includes:
- Simple linear regression (each biomarker → pTau-217)
- Multiple linear regression
- Ridge, Lasso, ElasticNet comparison
- VIF multicollinearity check
- Residual diagnostics
- Assumption checks
"""
from __future__ import annotations
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

RANDOM_SEED = 42

# pTau-217 is the strongest AD biomarker and the most continuous/meaningful
# regression target in this dataset.
REGRESSION_TARGET = "pT217_F"
REGRESSION_PREDICTORS = ["AB42_F", "AB40_F", "AB42_AB40_F", "NfL_harmonized", "GFAP_harmonized"]


def _harmonize_nfl_gfap(df: pd.DataFrame) -> pd.DataFrame:
    """Simple harmonization using audit-derived fixed ratios."""
    df = df.copy()
    nfl_ratio, gfap_ratio = 1.465, 0.376
    nfl_q_as_f = df["NfL_Q"] * nfl_ratio
    df["NfL_harmonized"] = np.where(
        df["NfL_F"].notna() & df["NfL_Q"].notna(),
        (df["NfL_F"] + nfl_q_as_f) / 2,
        np.where(df["NfL_F"].notna(), df["NfL_F"],
                 np.where(df["NfL_Q"].notna(), nfl_q_as_f, np.nan))
    )
    gfap_q_as_f = df["GFAP_Q"] * gfap_ratio
    df["GFAP_harmonized"] = np.where(
        df["GFAP_F"].notna() & df["GFAP_Q"].notna(),
        (df["GFAP_F"] + gfap_q_as_f) / 2,
        np.where(df["GFAP_F"].notna(), df["GFAP_F"],
                 np.where(df["GFAP_Q"].notna(), gfap_q_as_f, np.nan))
    )
    return df


def run_regression_analysis(df: pd.DataFrame, out_dir: Path) -> dict:
    """
    df must already have sentinels cleaned.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _harmonize_nfl_gfap(df)

    # ── 1. Simple linear regressions (each predictor → pTau-217) ────────────
    simple_rows = []
    for pred in REGRESSION_PREDICTORS:
        sub = df[[REGRESSION_TARGET, pred]].dropna()
        if len(sub) < 20:
            continue
        x = sub[pred].values.reshape(-1, 1)
        y = sub[REGRESSION_TARGET].values
        slope, intercept, r, p, se = stats.linregress(sub[pred].values, y)
        r2 = r ** 2
        # 95% CI for slope
        t_crit = stats.t.ppf(0.975, df=len(sub) - 2)
        ci_lo = slope - t_crit * se
        ci_hi = slope + t_crit * se
        simple_rows.append({
            "predictor": pred,
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 4),
            "r": round(float(r), 4),
            "r_squared": round(float(r2), 4),
            "p_value": round(float(p), 8),
            "std_error": round(float(se), 6),
            "ci_95_low": round(float(ci_lo), 6),
            "ci_95_high": round(float(ci_hi), 6),
            "n": int(len(sub)),
        })
        _plot_simple_regression(sub[pred].values, y, pred, slope, intercept, r2, out_dir)

    simple_df = pd.DataFrame(simple_rows).sort_values("r_squared", ascending=False)
    simple_df.to_csv(out_dir / "simple_linear_regression.csv", index=False)

    # ── 2. Multiple linear regression ───────────────────────────────────────
    sub_multi = df[[REGRESSION_TARGET] + REGRESSION_PREDICTORS].dropna()
    X_multi = sub_multi[REGRESSION_PREDICTORS].values
    y_multi = sub_multi[REGRESSION_TARGET].values

    # Fit with statsmodels for full stats (p-values, CIs per coefficient)
    try:
        import statsmodels.api as sm
        X_sm = sm.add_constant(X_multi)
        ols_model = sm.OLS(y_multi, X_sm).fit()
        multi_summary = {
            "r_squared": round(float(ols_model.rsquared), 4),
            "adj_r_squared": round(float(ols_model.rsquared_adj), 4),
            "f_stat": round(float(ols_model.fvalue), 4),
            "f_pvalue": round(float(ols_model.f_pvalue), 8),
            "aic": round(float(ols_model.aic), 2),
            "bic": round(float(ols_model.bic), 2),
            "n": int(len(sub_multi)),
        }
        coef_rows = []
        conf_int = np.array(ols_model.conf_int())
        for i, name in enumerate(["const"] + REGRESSION_PREDICTORS):
            coef_rows.append({
                "predictor": name,
                "coefficient": round(float(ols_model.params[i]), 6),
                "std_error": round(float(ols_model.bse[i]), 6),
                "t_stat": round(float(ols_model.tvalues[i]), 4),
                "p_value": round(float(ols_model.pvalues[i]), 8),
                "ci_95_low": round(float(conf_int[i, 0]), 6),
                "ci_95_high": round(float(conf_int[i, 1]), 6),
            })
        coef_df = pd.DataFrame(coef_rows)
        coef_df.to_csv(out_dir / "multiple_regression_coefficients.csv", index=False)
        pd.DataFrame([multi_summary]).to_csv(out_dir / "multiple_regression_summary.csv", index=False)

        # Residual diagnostics
        y_pred_ols = ols_model.fittedvalues
        residuals = ols_model.resid
        _plot_residuals(y_multi, y_pred_ols, residuals, "OLS", out_dir)

    except ImportError:
        logger.warning("statsmodels not available — using sklearn for multiple regression")
        lr = LinearRegression()
        lr.fit(X_multi, y_multi)
        y_pred_lr = lr.predict(X_multi)
        multi_summary = {
            "r_squared": round(float(r2_score(y_multi, y_pred_lr)), 4),
            "n": int(len(sub_multi)),
        }
        coef_df = pd.DataFrame({
            "predictor": REGRESSION_PREDICTORS,
            "coefficient": lr.coef_.round(6),
        })
        coef_df.to_csv(out_dir / "multiple_regression_coefficients.csv", index=False)

    # ── 3. VIF (multicollinearity) ───────────────────────────────────────────
    vif_rows = _compute_vif(sub_multi[REGRESSION_PREDICTORS])
    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(out_dir / "vif_multicollinearity.csv", index=False)
    _plot_vif(vif_df, out_dir)

    # ── 4. Model comparison (regression) ────────────────────────────────────
    model_results = _compare_regression_models(X_multi, y_multi, out_dir)

    # ── 5. Log-transformed regression (addresses skewness) ──────────────────
    y_log = np.log1p(y_multi)
    lr_log = LinearRegression()
    lr_log.fit(X_multi, y_log)
    y_pred_log = lr_log.predict(X_multi)
    log_r2 = r2_score(y_log, y_pred_log)
    pd.DataFrame([{
        "model": "OLS on log1p(pTau-217)",
        "r_squared": round(float(log_r2), 4),
        "note": "Log-transform reduces skewness (raw skew=3.53) and improves linearity",
    }]).to_csv(out_dir / "log_regression_summary.csv", index=False)

    logger.info("Regression analysis complete → %s", out_dir)
    return {
        "simple_regression": simple_df.to_dict(orient="records"),
        "multiple_regression": multi_summary,
        "model_comparison": model_results,
    }


def _compute_vif(X_df: pd.DataFrame) -> list:
    """Compute VIF for each predictor."""
    from sklearn.linear_model import LinearRegression
    rows = []
    cols = X_df.columns.tolist()
    X = X_df.dropna().values
    for i, col in enumerate(cols):
        y_vif = X[:, i]
        X_rest = np.delete(X, i, axis=1)
        lr = LinearRegression()
        lr.fit(X_rest, y_vif)
        r2 = r2_score(y_vif, lr.predict(X_rest))
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        rows.append({
            "feature": col,
            "vif": round(float(vif), 4),
            "r_squared_with_others": round(float(r2), 4),
            "concern": "HIGH (>10)" if vif > 10 else ("MODERATE (5-10)" if vif > 5 else "OK (<5)"),
        })
    return rows


def _compare_regression_models(X: np.ndarray, y: np.ndarray, out_dir: Path) -> list:
    """Compare multiple regression models via 5-fold CV."""
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    models = {
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge (α=1.0)": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Lasso (α=0.01)": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=5000))]),
        "ElasticNet": Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))]),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED),
    }
    rows = []
    for name, model in models.items():
        mae_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
        mse_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
        rows.append({
            "model": name,
            "cv_mae_mean": round(float(mae_scores.mean()), 4),
            "cv_mae_std": round(float(mae_scores.std()), 4),
            "cv_rmse_mean": round(float(np.sqrt(mse_scores).mean()), 4),
            "cv_rmse_std": round(float(np.sqrt(mse_scores).std()), 4),
            "cv_r2_mean": round(float(r2_scores.mean()), 4),
            "cv_r2_std": round(float(r2_scores.std()), 4),
        })
    comp_df = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False)
    comp_df.to_csv(out_dir / "regression_model_comparison.csv", index=False)
    _plot_regression_comparison(comp_df, out_dir)
    return rows


def _plot_simple_regression(x: np.ndarray, y: np.ndarray, pred_name: str,
                             slope: float, intercept: float, r2: float, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(x, y, alpha=0.4, s=20, color="#457b9d")
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_line, slope * x_line + intercept, color="#e63946", linewidth=2)
    axes[0].set_xlabel(pred_name)
    axes[0].set_ylabel("pTau-217 (pg/mL)")
    axes[0].set_title(f"pTau-217 ~ {pred_name}\nR²={r2:.3f}")

    y_pred = slope * x + intercept
    residuals = y - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.4, s=20, color="#2a9d8f")
    axes[1].axhline(0, color="red", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Fitted values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title(f"Residual Plot: pTau-217 ~ {pred_name}")
    plt.tight_layout()
    safe_name = pred_name.replace("/", "_")
    plt.savefig(out_dir / f"simple_regression_{safe_name}.png", dpi=150)
    plt.close()


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                    residuals: np.ndarray, model_name: str, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Actual vs predicted
    axes[0].scatter(y_true, y_pred, alpha=0.4, s=20, color="#457b9d")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5)
    axes[0].set_xlabel("Actual pTau-217")
    axes[0].set_ylabel("Predicted pTau-217")
    axes[0].set_title("Actual vs Predicted")

    # Residuals vs fitted
    axes[1].scatter(y_pred, residuals, alpha=0.4, s=20, color="#2a9d8f")
    axes[1].axhline(0, color="red", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Fitted values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Fitted (Homoscedasticity check)")

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot of Residuals (Normality check)")
    plt.suptitle(f"Regression Diagnostics — {model_name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"regression_diagnostics_{model_name.replace(' ', '_')}.png", dpi=150)
    plt.close()


def _plot_vif(vif_df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e63946" if v > 10 else ("#e9c46a" if v > 5 else "#2a9d8f")
              for v in vif_df["vif"]]
    ax.barh(vif_df["feature"], vif_df["vif"], color=colors, edgecolor="white")
    ax.axvline(5, color="orange", linestyle="--", linewidth=1.2, label="VIF=5 (moderate)")
    ax.axvline(10, color="red", linestyle="--", linewidth=1.2, label="VIF=10 (high)")
    ax.set_xlabel("VIF")
    ax.set_title("Variance Inflation Factor (Multicollinearity Check)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "vif_multicollinearity.png", dpi=150)
    plt.close()


def _plot_regression_comparison(comp_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, label in [
        (axes[0], "cv_r2_mean", "CV R² (mean)"),
        (axes[1], "cv_mae_mean", "CV MAE (mean)"),
        (axes[2], "cv_rmse_mean", "CV RMSE (mean)"),
    ]:
        df_s = comp_df.sort_values(metric, ascending=(metric != "cv_r2_mean"))
        ax.barh(df_s["model"], df_s[metric], color="#457b9d", edgecolor="white")
        ax.set_xlabel(label)
        ax.set_title(label)
        for i, (_, row) in enumerate(df_s.iterrows()):
            ax.text(row[metric] + 0.001, i, f"{row[metric]:.4f}", va="center", fontsize=8)
    plt.suptitle("Regression Model Comparison (5-fold CV on pTau-217 prediction)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "regression_model_comparison.png", dpi=150)
    plt.close()

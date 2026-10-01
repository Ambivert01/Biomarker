"""
run_analysis.py
================
Main entry point. Runs the complete statistical analysis pipeline on the
real ADNI dataset. Integrates with the existing project without modifying
any existing code.

Usage:
    cd /home/ambivert/Downloads/adni_project
    python3 run_analysis.py
"""
from __future__ import annotations
import logging
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "results" / "analysis.log", mode="w"),
    ],
)
logger = logging.getLogger("run_analysis")

# ── Output directories ───────────────────────────────────────────────────────
RESULTS = PROJECT_ROOT / "results"
DIRS = {
    "statistics":      RESULTS / "statistics",
    "correlations":    RESULTS / "correlations",
    "regression":      RESULTS / "regression",
    "classification":  RESULTS / "classification",
    "feature_analysis":RESULTS / "feature_analysis",
    "distributions":   RESULTS / "distributions",
    "outliers":        RESULTS / "outliers",
    "model_comparison":RESULTS / "model_comparison",
    "reports":         RESULTS / "reports",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR     = ARTIFACTS_DIR / "models"
DATA_DIR      = PROJECT_ROOT / "data"

# ── Sentinel cleaning (mirrors preprocessing pipeline) ───────────────────────
BIO_COLS = [
    "pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F",
    "NfL_Q", "GFAP_Q", "NfL_F", "GFAP_F",
]

def clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in BIO_COLS:
        if c in df.columns:
            df.loc[df[c] < 0, c] = float("nan")
    if "MMSE_FINAL_SCORE" in df.columns:
        df.loc[(df["MMSE_FINAL_SCORE"] < 0) | (df["MMSE_FINAL_SCORE"] > 30),
               "MMSE_FINAL_SCORE"] = float("nan")
    return df


def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("ADNI Plasma Biomarker — Complete Statistical Analysis Pipeline")
    logger.info("=" * 70)

    # ── Load dataset ─────────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    df_raw = pd.read_parquet(DATA_DIR / "Final_Biomarker_Patients.parquet")
    df = clean_sentinels(df_raw)
    logger.info("Dataset loaded: %d rows × %d cols", len(df), len(df.columns))
    logger.info("Class distribution: %s", df["GROUP"].value_counts().to_dict())

    all_results = {}

    # ── Step 1: Data Quality ─────────────────────────────────────────────────
    logger.info("\n[1/8] Running data quality analysis...")
    try:
        from analysis.data_quality import run_data_quality
        r = run_data_quality(df_raw, DIRS["distributions"])
        all_results["data_quality"] = r
        logger.info("  ✓ Data quality complete")
    except Exception as e:
        logger.error("  ✗ Data quality failed: %s", e, exc_info=True)

    # ── Step 2: Descriptive Statistics ───────────────────────────────────────
    logger.info("\n[2/8] Running descriptive statistics...")
    try:
        from analysis.descriptive_statistics import run_descriptive_statistics
        r = run_descriptive_statistics(df, DIRS["statistics"])
        all_results["descriptive_statistics"] = r
        logger.info("  ✓ Descriptive statistics complete")
    except Exception as e:
        logger.error("  ✗ Descriptive statistics failed: %s", e, exc_info=True)

    # ── Step 3: Correlation Analysis ─────────────────────────────────────────
    logger.info("\n[3/8] Running correlation analysis...")
    try:
        from analysis.correlation import run_correlation_analysis
        r = run_correlation_analysis(df, DIRS["correlations"])
        all_results["correlation"] = {"status": "complete"}
        logger.info("  ✓ Correlation analysis complete")
    except Exception as e:
        logger.error("  ✗ Correlation analysis failed: %s", e, exc_info=True)

    # ── Step 4: Statistical Tests ─────────────────────────────────────────────
    logger.info("\n[4/8] Running statistical tests...")
    try:
        from analysis.statistical_tests import run_statistical_tests
        r = run_statistical_tests(df, DIRS["statistics"])
        all_results["statistical_tests"] = r
        logger.info("  ✓ Statistical tests complete")
    except Exception as e:
        logger.error("  ✗ Statistical tests failed: %s", e, exc_info=True)

    # ── Step 5: Regression Analysis ──────────────────────────────────────────
    logger.info("\n[5/8] Running regression analysis...")
    try:
        from analysis.regression import run_regression_analysis
        r = run_regression_analysis(df, DIRS["regression"])
        all_results["regression"] = r
        logger.info("  ✓ Regression analysis complete")
    except Exception as e:
        logger.error("  ✗ Regression analysis failed: %s", e, exc_info=True)

    # ── Step 6: Classification Analysis ──────────────────────────────────────
    logger.info("\n[6/8] Running classification analysis...")
    try:
        from analysis.classification import run_classification_analysis
        r = run_classification_analysis(df, DIRS["classification"], MODEL_DIR)
        all_results["classification"] = r
        logger.info("  ✓ Classification analysis complete")
    except Exception as e:
        logger.error("  ✗ Classification analysis failed: %s", e, exc_info=True)

    # ── Step 7: Feature Analysis ──────────────────────────────────────────────
    logger.info("\n[7/8] Running feature analysis...")
    try:
        from analysis.feature_analysis import run_feature_analysis
        r = run_feature_analysis(df, DIRS["feature_analysis"], MODEL_DIR)
        all_results["feature_analysis"] = {"status": "complete"}
        logger.info("  ✓ Feature analysis complete")
    except Exception as e:
        logger.error("  ✗ Feature analysis failed: %s", e, exc_info=True)

    # ── Step 8: Outlier Analysis ──────────────────────────────────────────────
    logger.info("\n[8/8] Running outlier analysis...")
    try:
        from analysis.outlier_analysis import run_outlier_analysis
        r = run_outlier_analysis(df, DIRS["outliers"])
        all_results["outlier_analysis"] = r
        logger.info("  ✓ Outlier analysis complete")
    except Exception as e:
        logger.error("  ✗ Outlier analysis failed: %s", e, exc_info=True)

    # ── Step 9: Model Comparison ──────────────────────────────────────────────
    logger.info("\n[9/9] Running model comparison...")
    try:
        from analysis.model_comparison import run_model_comparison
        r = run_model_comparison(df, DIRS["model_comparison"], ARTIFACTS_DIR)
        all_results["model_comparison"] = r
        logger.info("  ✓ Model comparison complete")
    except Exception as e:
        logger.error("  ✗ Model comparison failed: %s", e, exc_info=True)

    # ── Save master results JSON ──────────────────────────────────────────────
    results_path = DIRS["reports"] / "analysis_summary.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("Analysis complete in %.1f seconds", elapsed)
    logger.info("Results saved to: %s", RESULTS)
    logger.info("=" * 70)

    _print_file_summary(RESULTS)


def _print_file_summary(results_dir: Path):
    files = sorted(results_dir.rglob("*.*"))
    logger.info("\nGenerated files (%d total):", len(files))
    for f in files:
        rel = f.relative_to(results_dir)
        logger.info("  results/%s", rel)


if __name__ == "__main__":
    main()

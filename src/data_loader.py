"""
data_loader.py
==============
Loads the ADNI Final_Biomarker_Patients sheet and runs schema / integrity
validation *every time* the pipeline runs (not just once, ad hoc, in a
notebook). If the source workbook ever changes, this module will raise
loudly instead of silently training on bad data.

This module encodes the findings of reports/DATA_AUDIT_REPORT.md as
executable checks.
"""
from __future__ import annotations
import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DataValidationError(Exception):
    """Raised when the raw data fails an integrity check that the pipeline depends on."""


@dataclass
class ValidationReport:
    n_rows: int = 0
    n_cols: int = 0
    checks_passed: list = field(default_factory=list)
    checks_failed: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def ok(self) -> bool:
        return len(self.checks_failed) == 0

    def summary(self) -> str:
        lines = [f"ValidationReport: rows={self.n_rows} cols={self.n_cols}",
                 f"  passed: {len(self.checks_passed)}  failed: {len(self.checks_failed)}  warnings: {len(self.warnings)}"]
        for c in self.checks_failed:
            lines.append(f"  [FAIL] {c}")
        for w in self.warnings:
            lines.append(f"  [WARN] {w}")
        return "\n".join(lines)


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load the Final_Biomarker_Patients sheet with the correct header row."""
    path = config["data"]["raw_excel_path"]
    sheet = config["data"]["sheet_name"]
    header_row = config["data"]["header_row"]
    df = pd.read_excel(path, sheet_name=sheet, header=header_row)
    logger.info(f"Loaded '{sheet}' from {path}: shape={df.shape}")

    # EXAMDATE and VISCODE arrive as raw strings (audit finding 3.x) -> parse dates explicitly.
    examdate_col = config["columns"]["examdate_column"]
    if examdate_col in df.columns:
        df[examdate_col] = pd.to_datetime(df[examdate_col], errors="coerce")

    return df


def validate_raw_data(df: pd.DataFrame, config: dict) -> ValidationReport:
    """
    Runs the integrity checks established in the audit. Hard failures raise;
    soft issues are recorded as warnings (they are handled downstream in
    preprocessing, but we still want them surfaced every run).
    """
    report = ValidationReport(n_rows=len(df), n_cols=len(df.columns))
    id_cols = config["data"]["id_columns"]
    target_col = config["data"]["target_column"]
    group_col = config["data"]["group_column"]
    class_labels = config["data"]["class_labels"]

    # --- structural checks -------------------------------------------------
    for col in id_cols + [target_col, group_col]:
        if col not in df.columns:
            report.checks_failed.append(f"Missing required column: {col}")
    if report.checks_failed:
        return report  # can't proceed further without core columns

    # one row per patient
    if df["RID"].duplicated().any():
        report.checks_failed.append("Duplicate RID found — Final_Biomarker_Patients must be one row/patient.")
    else:
        report.checks_passed.append("No duplicate RID (one row per patient).")

    if df["PTID"].duplicated().any():
        report.checks_failed.append("Duplicate PTID found.")
    else:
        report.checks_passed.append("No duplicate PTID.")

    if df.duplicated().any():
        report.checks_failed.append(f"{df.duplicated().sum()} fully duplicated rows found.")
    else:
        report.checks_passed.append("No fully duplicated rows.")

    # target validity
    valid_targets = set(class_labels.keys())
    bad_targets = ~df[target_col].isin(valid_targets)
    if bad_targets.any():
        report.checks_failed.append(f"{bad_targets.sum()} rows have DIAGNOSIS outside {valid_targets}.")
    else:
        report.checks_passed.append(f"All DIAGNOSIS values in expected set {valid_targets}.")

    # DIAGNOSIS <-> GROUP consistency
    expected_group = df[target_col].map(class_labels)
    mismatch = (expected_group != df[group_col]).sum()
    if mismatch:
        report.checks_failed.append(f"{mismatch} rows where GROUP text doesn't match DIAGNOSIS code.")
    else:
        report.checks_passed.append("DIAGNOSIS and GROUP are fully consistent.")

    # class balance sanity (not a hard failure, just informative)
    counts = df[group_col].value_counts()
    report.warnings.append(f"Class distribution: {counts.to_dict()}")

    # --- sentinel / range checks (soft — handled in preprocessing, just logged here) ---
    numeric_bio_cols = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in id_cols + [target_col]]
    for c in numeric_bio_cols:
        n_negative = (df[c] < 0).sum()
        if n_negative > 0:
            report.warnings.append(f"Column '{c}' has {n_negative} negative (sentinel) values — will be recoded to NaN in preprocessing.")

    mmse_col = config["columns"]["mmse_column"]
    if mmse_col in df.columns:
        lo, hi = config["sentinels"]["mmse_valid_range"]
        out_of_range = ((df[mmse_col] < lo) | (df[mmse_col] > hi)).sum()
        if out_of_range:
            report.warnings.append(f"MMSE has {out_of_range} out-of-range value(s) (outside [{lo},{hi}]) — will be recoded to NaN.")
        n_missing = df[mmse_col].isna().sum()
        report.warnings.append(f"MMSE missing (NaN): {n_missing}/{len(df)} ({100*n_missing/len(df):.1f}%).")

    # rare category check
    phase_col = config["columns"]["phase_column"]
    if phase_col in df.columns:
        phase_counts = df[phase_col].value_counts()
        rare = phase_counts[phase_counts < 10]
        if len(rare):
            report.warnings.append(f"Rare PHASE categories (<10 patients): {rare.to_dict()}")

    logger.info(report.summary())
    return report


def load_and_validate(config_path: str | Path = "../config/config.yaml") -> tuple[pd.DataFrame, ValidationReport]:
    config = load_config(config_path)
    df = load_raw_data(config)
    report = validate_raw_data(df, config)
    if not report.ok():
        raise DataValidationError("Raw data failed hard validation checks:\n" + report.summary())
    return df, report


if __name__ == "__main__":
    df, report = load_and_validate("../config/config.yaml")
    print(report.summary())
    print(df.head())

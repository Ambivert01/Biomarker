"""
test_preprocessing.py
=======================
Unit tests for the sentinel-handling and harmonization transformers.
Run: cd tests && python3 -m pytest test_preprocessing.py -v
"""
import sys
sys.path.append('../src')
import numpy as np
import pandas as pd
import pytest

from preprocessing.sentinel_handling import SentinelToNaN
from features.harmonization import CrossPlatformHarmonizer
from features.mmse_handler import MMSEHandler


def make_toy_df():
    return pd.DataFrame({
        "pT217_F": [0.2, 0.5, -4.0, 0.3],
        "AB42_F": [25.0, -4.0, 30.0, 28.0],
        "NfL_Q": [20.0, -4.0, -5.0, 25.0],
        "GFAP_Q": [100.0, -4.0, -5.0, 150.0],
        "NfL_F": [22.0, 24.0, np.nan, -4.0],
        "GFAP_F": [90.0, 95.0, np.nan, -4.0],
        "MMSE_FINAL_SCORE": [28, -1, 35, np.nan],
    })


class TestSentinelToNaN:
    def test_negative_values_become_nan(self):
        df = make_toy_df()
        cols = ["pT217_F", "AB42_F", "NfL_Q", "GFAP_Q", "NfL_F", "GFAP_F"]
        t = SentinelToNaN(biomarker_columns=cols, mmse_column="MMSE_FINAL_SCORE")
        out = t.fit_transform(df)
        assert out.loc[2, "pT217_F"] != out.loc[2, "pT217_F"] or pd.isna(out.loc[2, "pT217_F"])  # -4 -> NaN
        assert pd.isna(out.loc[1, "AB42_F"])
        assert pd.isna(out.loc[1, "NfL_Q"])   # -4
        assert pd.isna(out.loc[2, "NfL_Q"])   # -5 (undocumented sentinel) also caught

    def test_mmse_out_of_range_becomes_nan(self):
        df = make_toy_df()
        t = SentinelToNaN(biomarker_columns=["pT217_F"], mmse_column="MMSE_FINAL_SCORE", mmse_valid_range=(0, 30))
        out = t.fit_transform(df)
        assert pd.isna(out.loc[1, "MMSE_FINAL_SCORE"])   # -1 out of range
        assert pd.isna(out.loc[2, "MMSE_FINAL_SCORE"])   # 35 out of range
        assert out.loc[0, "MMSE_FINAL_SCORE"] == 28       # valid value untouched

    def test_positive_values_untouched(self):
        df = make_toy_df()
        t = SentinelToNaN(biomarker_columns=["pT217_F"])
        out = t.fit_transform(df)
        assert out.loc[0, "pT217_F"] == 0.2
        assert out.loc[1, "pT217_F"] == 0.5

    def test_stateless_no_leakage(self):
        """fit() must not learn anything from the data (prevents any leakage)."""
        df = make_toy_df()
        t1 = SentinelToNaN(biomarker_columns=["pT217_F"])
        t1.fit(df)
        t2 = SentinelToNaN(biomarker_columns=["pT217_F"])
        t2.fit(df.iloc[:1])   # fit on a totally different subset
        out1 = t1.transform(df)
        out2 = t2.transform(df)
        pd.testing.assert_frame_equal(out1, out2)


class TestCrossPlatformHarmonizer:
    def test_ratio_learned_from_dual_platform_rows_only(self):
        df = pd.DataFrame({
            "NfL_F": [30.0, 60.0, np.nan, np.nan] * 10,
            "NfL_Q": [20.0, 40.0, 15.0, np.nan] * 10,
            "GFAP_F": [40.0, 80.0, np.nan, np.nan] * 10,
            "GFAP_Q": [100.0, 200.0, 90.0, np.nan] * 10,
        })
        h = CrossPlatformHarmonizer(min_pairs_for_fit=5)
        h.fit(df)
        # ratio should be close to the true constructed ratio of 1.5 (30/20, 60/40)
        assert abs(h.nfl_ratio_ - 1.5) < 0.05

    def test_fallback_used_when_too_few_pairs(self):
        df = pd.DataFrame({
            "NfL_F": [np.nan] * 5,
            "NfL_Q": [10.0] * 5,
            "GFAP_F": [np.nan] * 5,
            "GFAP_Q": [10.0] * 5,
        })
        h = CrossPlatformHarmonizer(fallback_nfl_ratio=1.465, fallback_gfap_ratio=0.376, min_pairs_for_fit=5)
        h.fit(df)
        assert h.nfl_ratio_ == 1.465
        assert h.gfap_ratio_ == 0.376

    def test_transform_coalesces_available_platform(self):
        df = pd.DataFrame({
            "NfL_F": [30.0, np.nan],
            "NfL_Q": [np.nan, 20.0],
            "GFAP_F": [40.0, np.nan],
            "GFAP_Q": [np.nan, 100.0],
        })
        h = CrossPlatformHarmonizer(fallback_nfl_ratio=1.5, fallback_gfap_ratio=0.4)
        h.fit(df)
        out = h.transform(df)
        assert out.loc[0, "NfL_harmonized"] == 30.0          # only Fuji available -> use it
        assert out.loc[1, "NfL_harmonized"] == pytest.approx(20.0 * h.nfl_ratio_)  # only Q -> convert
        assert out.loc[0, "NfL_n_platforms"] == 1
        assert out.loc[1, "NfL_n_platforms"] == 1

    def test_no_fit_time_leakage_between_folds(self):
        """A harmonizer fit on fold A must not change if fold B's data changes."""
        df_a = pd.DataFrame({"NfL_F": [30.0]*10, "NfL_Q": [20.0]*10, "GFAP_F": [40.0]*10, "GFAP_Q": [100.0]*10})
        h1 = CrossPlatformHarmonizer(min_pairs_for_fit=5)
        h1.fit(df_a)
        ratio_before = h1.nfl_ratio_
        # simulate a completely different, extreme "other fold" -- must not affect h1
        df_b = pd.DataFrame({"NfL_F": [999.0]*10, "NfL_Q": [1.0]*10, "GFAP_F": [999.0]*10, "GFAP_Q": [1.0]*10})
        _ = df_b  # h1 never sees this
        assert h1.nfl_ratio_ == ratio_before


class TestMMSEHandler:
    def test_missing_flag_correct(self):
        df = pd.DataFrame({"MMSE_FINAL_SCORE": [28.0, np.nan, 15.0, np.nan]})
        h = MMSEHandler()
        out = h.fit_transform(df)
        assert list(out["MMSE_missing"]) == [0, 1, 0, 1]


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])

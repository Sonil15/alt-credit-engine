"""Cohort-aware imputation: missing sources fill to typical-applicant, not to a
directionally biased zero, while structurally-N/A features stay zero."""

import numpy as np
import pandas as pd

from models_ai import imputation
from models_ai.constants import FEATURE_COLUMNS, fill_missing_features


# cohort_code 0 = an income-bearing cohort with a defined income median;
# cohort_code 9 = a cohort for which the business feature is structurally absent.
STATS = {
    "global": {"monthly_income_mean": 50000.0, "business_vintage_years": 6.0},
    "by_cohort": {
        "0": {"monthly_income_mean": 80000.0, "business_vintage_years": None},
        "3": {"monthly_income_mean": 40000.0, "business_vintage_years": 7.0},
    },
}


def _use_test_stats(monkeypatch):
    monkeypatch.setattr(imputation, "_CACHE", STATS)


def test_missing_source_fills_cohort_median_not_zero(monkeypatch):
    _use_test_stats(monkeypatch)
    row = pd.DataFrame([{"cohort_code": 0.0, "monthly_income_mean": np.nan}])
    filled = fill_missing_features(row)
    # A salaried applicant missing cashflow gets the salaried cohort's typical income,
    # not 0 (which would read as "no income" and unfairly punish them).
    assert filled["monthly_income_mean"].iloc[0] == 80000.0


def test_structurally_na_feature_stays_zero(monkeypatch):
    _use_test_stats(monkeypatch)
    row = pd.DataFrame([{"cohort_code": 0.0}])  # no business columns at all
    filled = fill_missing_features(row)
    # business vintage is undefined for this cohort -> "no business" -> 0.0,
    # never the global median (6.0).
    assert filled["business_vintage_years"].iloc[0] == 0.0


def test_per_row_cohort_in_mixed_batch(monkeypatch):
    _use_test_stats(monkeypatch)
    batch = pd.DataFrame(
        [
            {"cohort_code": 0.0, "monthly_income_mean": np.nan},
            {"cohort_code": 3.0, "monthly_income_mean": np.nan},
        ]
    )
    filled = fill_missing_features(batch).reset_index(drop=True)
    # Each row imputes from its own cohort, not a single shared profile.
    assert filled["monthly_income_mean"].iloc[0] == 80000.0
    assert filled["monthly_income_mean"].iloc[1] == 40000.0


def test_unknown_cohort_falls_back_to_global(monkeypatch):
    _use_test_stats(monkeypatch)
    row = pd.DataFrame([{"cohort_code": np.nan, "monthly_income_mean": np.nan}])
    filled = fill_missing_features(row)
    assert filled["monthly_income_mean"].iloc[0] == 50000.0


def test_empty_profile_reverts_to_zero_fill(monkeypatch):
    # Before the artifact exists, behaviour is the historical zero-fill.
    monkeypatch.setattr(imputation, "_CACHE", {"global": {}, "by_cohort": {}})
    row = pd.DataFrame([{"cohort_code": 0.0, "monthly_income_mean": np.nan}])
    filled = fill_missing_features(row)
    assert filled["monthly_income_mean"].iloc[0] == 0.0
    assert list(filled.columns) == FEATURE_COLUMNS

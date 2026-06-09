import json
import math

import pandas as pd

from convergence.scorecard import compute_factor_points
from core.json_utils import safe_float


def test_safe_float_replaces_nan_and_inf():
    assert safe_float(float("nan")) == 0.0
    assert safe_float(float("inf")) == 0.0
    assert safe_float(float("-inf")) == 0.0
    assert safe_float("bad", default=0.5) == 0.5
    assert safe_float(0.42) == 0.42


def test_factor_points_are_json_serializable_with_nan_features():
    row = pd.Series(
        {
            "missed_payments_count": float("nan"),
            "monthly_income_mean": float("inf"),
            "resilience_coefficient": 0.8,
        }
    )
    points = compute_factor_points(row)
    assert all(math.isfinite(value) for value in points.values())
    json.dumps(points)

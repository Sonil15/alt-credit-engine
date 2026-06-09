import json
import math

import numpy as np
import pandas as pd

from convergence.scorecard import compute_factor_points
from core.json_utils import safe_float, safe_round, sanitize_for_json
from models.pydantic_schemas import CreditScoreResponse


def test_safe_float_replaces_nan_and_inf():
    assert safe_float(float("nan")) == 0.0
    assert safe_float(float("inf")) == 0.0
    assert safe_float(float("-inf")) == 0.0
    assert safe_float("bad", default=0.5) == 0.5
    assert safe_float(0.42) == 0.42
    assert safe_float(np.float64("nan")) == 0.0
    assert safe_float(np.float64("inf")) == 0.0


def test_safe_round_replaces_non_finite_values():
    assert safe_round(float("nan"), 4) == 0.0
    assert safe_round(np.float64("inf"), 2) == 0.0


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
    json.dumps(points, allow_nan=False)


def test_sanitize_for_json_handles_nested_numpy_values():
    payload = {
        "probability_of_default": np.float64("nan"),
        "shap_drivers": [{"feature": "x", "shap_value": np.float64("inf")}],
        "factor_points": {"monthly_income_mean": np.float64("nan")},
    }
    sanitized = sanitize_for_json(payload)
    json.dumps(sanitized, allow_nan=False)


def test_credit_score_response_serializes_with_strict_json_encoder():
    response = CreditScoreResponse(
        user_id="user-1",
        credit_score=600,
        probability_of_default=float("nan"),
        decision="REVIEW",
        auto_reject=False,
        shap_drivers=[{"feature": "missed_payments_count", "shap_value": float("inf")}],
        factor_points={"monthly_income_mean": float("nan")},
    )
    encoded = json.dumps(response.model_dump(), allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded

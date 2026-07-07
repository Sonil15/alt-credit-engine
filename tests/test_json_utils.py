import json
import math

import numpy as np
import pandas as pd

from convergence.scorecard import ebm_to_points
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


def test_ebm_points_are_finite_and_serializable():
    raw_contrib = [float("nan"), float("inf"), 0.42, -0.31]
    points = {f"f{i}": safe_round(ebm_to_points(safe_float(v)), 2) for i, v in enumerate(raw_contrib)}
    assert all(math.isfinite(value) for value in points.values())
    json.dumps(points, allow_nan=False)


def test_sanitize_for_json_handles_nested_numpy_values():
    payload = {
        "probability_of_default": np.float64("nan"),
        "feature_drivers": [{"feature": "x", "contribution_value": np.float64("inf")}],
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
        feature_drivers=[{"feature": "missed_payments_count", "contribution_value": float("inf")}],
        factor_points={"monthly_income_mean": float("nan")},
    )
    encoded = json.dumps(response.model_dump(), allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded

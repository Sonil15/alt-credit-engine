from convergence.reason_codes import format_reason_codes, shap_to_reason_codes


def test_shap_to_reason_codes():
    drivers = [
        {"feature": "missed_payments_count", "shap_value": 0.5},
        {"feature": "resilience_coefficient", "shap_value": -0.3},
    ]
    codes = shap_to_reason_codes(drivers)
    assert len(codes) == 1
    assert "Missed" in codes[0] or "missed" in codes[0].lower()


def test_format_reason_codes_empty():
    assert "No adverse" in format_reason_codes([])

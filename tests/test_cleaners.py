from preprocessing.clean_cashflow import clean_cashflow
from preprocessing.clean_ecommerce import clean_ecommerce
from preprocessing.clean_geo import clean_geo
from preprocessing.clean_telecom import clean_telecom


def test_clean_telecom_returns_series():
    invoices = [
        {
            "due_date": "2024-01-15",
            "payment_date": "2024-01-14",
            "status": "paid",
        },
        {
            "due_date": "2024-02-15",
            "payment_date": "2024-02-20",
            "status": "late",
        },
    ]
    features, series = clean_telecom(invoices)
    assert "avg_days_late" in features
    assert "missed_payments_count" in features
    assert isinstance(series, list)


def test_clean_cashflow_returns_monthly_series():
    txns = [
        {"txn_date": "2024-01-05", "type": "CREDIT", "amount": 50000, "narration": "NEFT/SALARY/ABC"},
        {"txn_date": "2024-01-10", "type": "DEBIT", "amount": 5000, "narration": "UPI/P2A/XYZ/Grocery"},
        {"txn_date": "2024-02-05", "type": "CREDIT", "amount": 48000, "narration": "NEFT/SALARY/DEF"},
    ]
    features, series = clean_cashflow(txns)
    assert features["monthly_income_mean"] > 0
    assert len(series) >= 1


def test_clean_ecommerce_necessity_ratio():
    orders = [
        {
            "item_category": "groceries",
            "amount": 1000,
            "merchant_rating_at_purchase": 4.5,
            "timestamp": "2024-03-01T10:00:00",
        },
        {
            "item_category": "luxury",
            "amount": 5000,
            "merchant_rating_at_purchase": 3.0,
            "timestamp": "2024-03-15T10:00:00",
        },
    ]
    features = clean_ecommerce(orders)
    assert 0 <= features["necessity_ratio"] <= 1


def test_clean_geo_spatial_variance():
    locations = [
        {"lat": 19.076, "long": 72.877},
        {"lat": 19.077, "long": 72.878},
        {"lat": 19.075, "long": 72.876},
    ]
    features = clean_geo(locations)
    assert "spatial_variance_score" in features
    assert features["anchor_count"] >= 1

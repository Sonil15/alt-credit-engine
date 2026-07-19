from collections import defaultdict
from datetime import datetime
from typing import Any

from preprocessing.clean_geo import historical_spatial_variance

NECESSITY_CATEGORIES = {"groceries", "utilities", "health", "education", "household"}
DISCRETIONARY_CATEGORIES = {"luxury", "electronics", "entertainment", "travel", "fashion"}


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def clean_ecommerce(raw_data: list[dict[str, Any]], sms_records: list[dict[str, Any]] | None = None) -> dict[str, float]:
    """Extract spending behavior features from e-commerce order records."""
    if not raw_data:
        return {
            "necessity_ratio": 0.0,
            "avg_merchant_rating": 0.0,
            "monthly_spend_volatility": 0.0,
            "historical_spatial_variance": 0.0,
            "distinct_pin_codes": 0.0,
            "sms_spend_total": 0.0,
        }

    necessity_spend = 0.0
    discretionary_spend = 0.0
    ratings: list[float] = []
    monthly_totals: dict[str, float] = defaultdict(float)

    for order in raw_data:
        category = str(order.get("item_category", "")).lower()
        amount = float(order.get("amount", 0.0))
        rating = order.get("merchant_rating_at_purchase")

        if category in NECESSITY_CATEGORIES:
            necessity_spend += amount
        elif category in DISCRETIONARY_CATEGORIES:
            discretionary_spend += amount
        else:
            necessity_spend += amount * 0.5
            discretionary_spend += amount * 0.5

        if rating is not None:
            ratings.append(float(rating))

        ts = _parse_timestamp(order.get("timestamp"))
        if ts is not None:
            month_key = ts.strftime("%Y-%m")
            monthly_totals[month_key] += amount

    total_spend = necessity_spend + discretionary_spend
    necessity_ratio = necessity_spend / total_spend if total_spend > 0 else 0.0
    avg_rating = sum(ratings) / len(ratings) if ratings else 0.0

    monthly_values = list(monthly_totals.values())
    if len(monthly_values) > 1:
        mean_spend = sum(monthly_values) / len(monthly_values)
        variance = sum((v - mean_spend) ** 2 for v in monthly_values) / len(monthly_values)
        volatility = variance ** 0.5
    else:
        volatility = 0.0

    sms_spend_total = 0.0
    if sms_records:
        import re
        amount_pat = re.compile(r"(?:rs\.?|inr|amt)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
        for s in sms_records:
            body = str(s.get("body", "")).lower()
            if any(k in body for k in ["debited", "spent", "charged", "purchase of", "paid to"]):
                match = amount_pat.search(body)
                if match:
                    try:
                        amt_str = match.group(1).replace(",", "")
                        sms_spend_total += float(amt_str)
                    except ValueError:
                        continue

    features = {
        "necessity_ratio": float(necessity_ratio),
        "avg_merchant_rating": float(avg_rating),
        "monthly_spend_volatility": float(volatility),
        "sms_spend_total": float(sms_spend_total),
    }
    features.update(historical_spatial_variance(raw_data))
    return features

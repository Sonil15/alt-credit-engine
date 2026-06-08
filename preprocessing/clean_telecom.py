from datetime import date
from typing import Any


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    return None


def clean_telecom(raw_data: list[dict[str, Any]]) -> tuple[dict[str, float], list[float]]:
    """Extract payment consistency features and monthly on-time payment rate series."""
    days_late: list[int] = []
    missed_payments = 0
    monthly_scores: dict[str, list[int]] = {}

    for record in raw_data:
        status = str(record.get("status", "")).lower()
        due_date = _parse_date(record.get("due_date"))
        month_key = due_date.strftime("%Y-%m") if due_date else None

        if status in {"defaulted", "missed", "unpaid"}:
            missed_payments += 1
            if month_key:
                monthly_scores.setdefault(month_key, []).append(0)
            continue

        payment_date = _parse_date(record.get("payment_date"))

        if due_date is None:
            continue

        if payment_date is None:
            if status == "late":
                missed_payments += 1
                if month_key:
                    monthly_scores.setdefault(month_key, []).append(0)
            continue

        delta = (payment_date - due_date).days
        days_late.append(max(delta, 0))
        if month_key:
            monthly_scores.setdefault(month_key, []).append(1 if delta <= 3 else 0)

    avg_days_late = sum(days_late) / len(days_late) if days_late else 0.0
    payment_series = [
        sum(scores) / len(scores) for _, scores in sorted(monthly_scores.items())
    ]

    return {
        "avg_days_late": float(avg_days_late),
        "missed_payments_count": float(missed_payments),
    }, payment_series

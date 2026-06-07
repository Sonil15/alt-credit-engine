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


def clean_telecom(raw_data: list[dict[str, Any]]) -> dict[str, float]:
    """Extract payment consistency features from telecom invoice records."""
    days_late: list[int] = []
    missed_payments = 0

    for record in raw_data:
        status = str(record.get("status", "")).lower()
        if status in {"defaulted", "missed", "unpaid"}:
            missed_payments += 1
            continue

        due_date = _parse_date(record.get("due_date"))
        payment_date = _parse_date(record.get("payment_date"))

        if due_date is None:
            continue

        if payment_date is None:
            if status == "late":
                missed_payments += 1
            continue

        delta = (payment_date - due_date).days
        days_late.append(max(delta, 0))

    avg_days_late = sum(days_late) / len(days_late) if days_late else 0.0

    return {
        "avg_days_late": float(avg_days_late),
        "missed_payments_count": float(missed_payments),
    }

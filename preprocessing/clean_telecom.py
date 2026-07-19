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


def clean_telecom(raw_data: list[dict[str, Any]], sms_records: list[dict[str, Any]] | None = None) -> tuple[dict[str, float], list[float]]:
    """Extract payment consistency features and monthly on-time payment rate series."""
    days_late: list[int] = []
    missed_payments = 0.0
    monthly_scores: dict[str, list[int]] = {}

    recharge_delays: list[int] = []
    sim_vintages: list[int] = []

    for record in raw_data:
        # Extract prepaid fields if present
        rd = record.get("recharge_delay_days")
        if rd is not None:
            recharge_delays.append(int(rd))

        sv = record.get("sim_vintage_months")
        if sv is not None:
            sim_vintages.append(int(sv))

        status = str(record.get("status", "")).lower()
        due_date = _parse_date(record.get("due_date"))
        month_key = due_date.strftime("%Y-%m") if due_date else None

        if status in {"defaulted", "missed", "unpaid"}:
            missed_payments += 1.0
            if month_key:
                monthly_scores.setdefault(month_key, []).append(0)
            continue

        payment_date = _parse_date(record.get("payment_date"))

        if due_date is None:
            continue

        if payment_date is None:
            if status == "late":
                missed_payments += 1.0
                if month_key:
                    monthly_scores.setdefault(month_key, []).append(0)
            continue

        delta = (payment_date - due_date).days
        days_late.append(max(delta, 0))
        if month_key:
            monthly_scores.setdefault(month_key, []).append(1 if delta <= 3 else 0)

    # Blend prepaid recharge delays into days_late
    if recharge_delays:
        days_late.extend(recharge_delays)
        # Missed recharges count as missed payments if delay > 5 days
        for rd in recharge_delays:
            if rd > 5:
                missed_payments += 1.0

    # Apply SIM vintage penalty if tenure is short (< 12 months)
    if sim_vintages:
        vintage = float(sim_vintages[-1])
        if vintage < 12.0:
            missed_payments += (12.0 - vintage) / 2.0

    avg_days_late = sum(days_late) / len(days_late) if days_late else 0.0
    payment_series = [
        sum(scores) / len(scores) for _, scores in sorted(monthly_scores.items())
    ]

    # Calculate sms_bill_delay from transactional SMS alerts
    sms_delays = []
    if sms_records:
        import datetime as dt
        sorted_sms = []
        for s in sms_records:
            t_val = s.get("timestamp")
            if t_val:
                try:
                    t_dt = dt.datetime.fromisoformat(str(t_val).replace("Z", "+00:00"))
                    sorted_sms.append((t_dt, str(s.get("sender", "")).upper(), str(s.get("body", "")).lower()))
                except Exception:
                    continue
        sorted_sms.sort(key=lambda x: x[0])

        alerts = []
        for t_dt, sender, body in sorted_sms:
            if any(k in body for k in ["due", "bill of", "invoice"]):
                alerts.append((t_dt, sender))
            elif any(k in body for k in ["paid", "received", "thank you"]):
                # Find the latest alert from the same sender group
                matching_alert = None
                for alert_dt, alert_sender in reversed(alerts):
                    # Compare sender first 3 characters to match JIOMOB/JIOPAY or BSCOM/BSPAY
                    if alert_sender[:3] == sender[:3]:
                        matching_alert = alert_dt
                        break
                if matching_alert:
                    delay = (t_dt - matching_alert).days
                    sms_delays.append(max(0, delay))

    sms_bill_delay = sum(sms_delays) / len(sms_delays) if sms_delays else 0.0

    return {
        "avg_days_late": float(avg_days_late),
        "missed_payments_count": float(missed_payments),
        "sms_bill_delay": float(sms_bill_delay),
    }, payment_series

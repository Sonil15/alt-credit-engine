import re
from datetime import date, datetime
from typing import Any

import pandas as pd

INCOME_KEYWORDS = re.compile(
    r"\b(salary|credit|refund|interest|dividend|cashback|deposit)\b",
    re.IGNORECASE,
)
EXPENSE_KEYWORDS = re.compile(
    r"\b(grocery|groceries|rent|utility|utilities|fuel|food|upi|p2a|p2p|purchase|bill)\b",
    re.IGNORECASE,
)
TRANSFER_KEYWORDS = re.compile(r"\b(transfer|neft|imps|rtgs|self)\b", re.IGNORECASE)


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    return None


def _categorize_transaction(txn_type: str, narration: str) -> str:
    txn_type_upper = txn_type.upper()
    narration_text = narration or ""

    if txn_type_upper == "CREDIT" or INCOME_KEYWORDS.search(narration_text):
        return "INCOME"
    if txn_type_upper == "DEBIT" or EXPENSE_KEYWORDS.search(narration_text):
        return "EXPENSE"
    if TRANSFER_KEYWORDS.search(narration_text):
        return "TRANSFER"
    return "EXPENSE" if txn_type_upper == "DEBIT" else "INCOME"


def clean_cashflow(raw_data: list[dict[str, Any]]) -> tuple[dict[str, float], list[float]]:
    """Parse cashflow transactions and return aggregate features plus monthly net-cashflow series."""
    empty = {
        "monthly_income_mean": 0.0,
        "monthly_expense_mean": 0.0,
        "cashflow_volatility": 0.0,
        "cash_burn_rate": 0.0,
        "upi_lite_txn_count": 0.0,
        "upi_lite_average_ticket": 0.0,
        "dbt_income_consistency": 0.0,
    }
    if not raw_data:
        return empty, []

    # Parse UPI Lite and DBT transactions
    upi_lite_amounts = []
    dbt_months = set()
    all_months = set()

    rows: list[dict[str, Any]] = []
    for txn in raw_data:
        txn_date = _parse_date(txn.get("txn_date"))
        if txn_date is None:
            continue

        month_key = txn_date.strftime("%Y-%m")
        all_months.add(month_key)

        category = _categorize_transaction(str(txn.get("type", "")), str(txn.get("narration", "")))
        amount = float(txn.get("amount", 0.0))
        signed_amount = amount if category == "INCOME" else -amount

        narration = str(txn.get("narration", "")).upper()
        txn_type = str(txn.get("type", "")).upper()

        # UPI Lite detection
        if "UPI-LITE" in narration or "UPI/LITE" in narration or "LITE-WALLET" in narration:
            upi_lite_amounts.append(amount)

        # DBT detection
        if category == "INCOME" and any(k in narration for k in ["DBT/", "APBS/", "PAHAL/", "PM-KISAN/", "PMKISAN/"]):
            dbt_months.add(month_key)

        rows.append(
            {
                "txn_date": txn_date,
                "category": category,
                "amount": amount,
                "signed_amount": signed_amount,
            }
        )

    if not rows:
        return empty, []

    upi_lite_txn_count = len(upi_lite_amounts)
    upi_lite_average_ticket = sum(upi_lite_amounts) / upi_lite_txn_count if upi_lite_txn_count > 0 else 0.0
    total_months = len(all_months) if all_months else 1
    dbt_income_consistency = len(dbt_months) / total_months

    # Calculate cash burn rate: ratio of expenses in first 7 days post-payday to payday credit amount
    from collections import defaultdict
    monthly_credits = defaultdict(list)
    for r in rows:
        if r["category"] == "INCOME":
            dt = r["txn_date"]
            month_key = dt.strftime("%Y-%m")
            monthly_credits[month_key].append((dt, r["amount"]))

    paydays = {}
    for m, credits in monthly_credits.items():
        if credits:
            # Pick the largest credit as the main paycheck/deposit for the month
            credits.sort(key=lambda x: x[1], reverse=True)
            paydays[m] = credits[0]  # (payday_date, credit_amount)

    burn_ratios = []
    for m, (payday_date, credit_amount) in paydays.items():
        if credit_amount <= 0.0:
            continue
        expense_in_window = 0.0
        for r in rows:
            if r["category"] == "EXPENSE":
                dt = r["txn_date"]
                delta = (dt - payday_date).days
                if 0 <= delta <= 7:
                     expense_in_window += r["amount"]
        ratio = expense_in_window / credit_amount
        burn_ratios.append(min(1.0, max(0.0, ratio)))

    cash_burn_rate = sum(burn_ratios) / len(burn_ratios) if burn_ratios else 0.0

    df = pd.DataFrame(rows)
    df["txn_date"] = pd.to_datetime(df["txn_date"])
    df = df.set_index("txn_date")

    weekly_net = df["signed_amount"].resample("W").sum().fillna(0.0)
    monthly_income = df.loc[df["category"] == "INCOME", "amount"].resample("ME").sum().fillna(0.0)
    monthly_expense = df.loc[df["category"] == "EXPENSE", "amount"].resample("ME").sum().fillna(0.0)
    monthly_net = monthly_income.reindex(monthly_expense.index.union(monthly_income.index), fill_value=0.0)
    monthly_net = monthly_net.subtract(monthly_expense.reindex(monthly_net.index, fill_value=0.0), fill_value=0.0)
    monthly_net = monthly_net.sort_index()

    income_mean = float(monthly_income.mean()) if not monthly_income.empty else 0.0
    expense_mean = float(monthly_expense.mean()) if not monthly_expense.empty else 0.0
    volatility = float(weekly_net.std()) if len(weekly_net) > 1 else 0.0

    features = {
        "monthly_income_mean": income_mean,
        "monthly_expense_mean": expense_mean,
        "cashflow_volatility": volatility if not pd.isna(volatility) else 0.0,
        "cash_burn_rate": float(cash_burn_rate),
        "upi_lite_txn_count": float(upi_lite_txn_count),
        "upi_lite_average_ticket": float(upi_lite_average_ticket),
        "dbt_income_consistency": float(dbt_income_consistency),
    }
    series = [float(v) for v in monthly_net.values.tolist()]
    return features, series

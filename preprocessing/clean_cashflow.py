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


def clean_cashflow(raw_data: list[dict[str, Any]]) -> dict[str, float]:
    """Parse cashflow transactions and resample into weekly statsmodels-ready features."""
    if not raw_data:
        return {
            "monthly_income_mean": 0.0,
            "monthly_expense_mean": 0.0,
            "cashflow_volatility": 0.0,
        }

    rows: list[dict[str, Any]] = []
    for txn in raw_data:
        txn_date = _parse_date(txn.get("txn_date"))
        if txn_date is None:
            continue

        category = _categorize_transaction(str(txn.get("type", "")), str(txn.get("narration", "")))
        amount = float(txn.get("amount", 0.0))
        signed_amount = amount if category == "INCOME" else -amount

        rows.append(
            {
                "txn_date": txn_date,
                "category": category,
                "amount": amount,
                "signed_amount": signed_amount,
            }
        )

    if not rows:
        return {
            "monthly_income_mean": 0.0,
            "monthly_expense_mean": 0.0,
            "cashflow_volatility": 0.0,
        }

    df = pd.DataFrame(rows)
    df["txn_date"] = pd.to_datetime(df["txn_date"])
    df = df.set_index("txn_date")

    weekly_net = df["signed_amount"].resample("W").sum().fillna(0.0)
    monthly_income = df.loc[df["category"] == "INCOME", "amount"].resample("ME").sum().fillna(0.0)
    monthly_expense = df.loc[df["category"] == "EXPENSE", "amount"].resample("ME").sum().fillna(0.0)

    income_mean = float(monthly_income.mean()) if not monthly_income.empty else 0.0
    expense_mean = float(monthly_expense.mean()) if not monthly_expense.empty else 0.0
    volatility = float(weekly_net.std()) if len(weekly_net) > 1 else 0.0

    return {
        "monthly_income_mean": income_mean,
        "monthly_expense_mean": expense_mean,
        "cashflow_volatility": volatility if not pd.isna(volatility) else 0.0,
    }

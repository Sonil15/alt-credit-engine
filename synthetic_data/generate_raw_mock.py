"""Generate 100 realistic mock user profiles for all alternative data sources."""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

fake = Faker("en_IN")

OUTPUT_PATH = Path(__file__).parent / "mock_data_100_users.json"

TELECOM_STATUSES = ["paid", "paid", "paid", "late", "defaulted", "missed"]
ECOMMERCE_CATEGORIES = [
    "electronics",
    "groceries",
    "luxury",
    "fashion",
    "travel",
    "utilities",
    "health",
    "education",
    "household",
    "entertainment",
]
RISK_APPETITES = ["low", "medium", "high"]
SAVINGS_FREQS = ["weekly", "monthly", "quarterly", "rarely", "never"]

INDIA_BOUNDS = {
    "lat_min": 8.0,
    "lat_max": 37.0,
    "long_min": 68.0,
    "long_max": 97.0,
}

NARRATION_TEMPLATES = [
    "UPI/P2A/{ref}/Grocery",
    "UPI/P2P/{ref}/Rent",
    "NEFT/SALARY/{ref}",
    "IMPS/TRANSFER/{ref}/Self",
    "UPI/MERCHANT/{ref}/Fuel",
    "DEBIT/ATM/WDL/{ref}",
    "CREDIT/CASHBACK/{ref}",
    "UPI/BILLPAY/{ref}/Electricity",
]


def generate_telecom_invoices(count: int = 12) -> list[dict]:
    invoices = []
    base_date = fake.date_between(start_date="-1y", end_date="-1m")

    for i in range(count):
        invoice_date = base_date + timedelta(days=30 * i)
        due_date = invoice_date + timedelta(days=15)
        status = random.choice(TELECOM_STATUSES)
        payment_date = None

        if status == "paid":
            payment_date = due_date + timedelta(days=random.randint(-2, 3))
        elif status == "late":
            payment_date = due_date + timedelta(days=random.randint(5, 20))
        elif status in {"defaulted", "missed", "unpaid"}:
            payment_date = None

        invoices.append(
            {
                "invoice_date": invoice_date.isoformat(),
                "due_date": due_date.isoformat(),
                "payment_date": payment_date.isoformat() if payment_date else None,
                "billed_amount": round(random.uniform(299, 1499), 2),
                "status": status,
            }
        )

    return invoices


def generate_ecommerce_orders(count: int = 20) -> list[dict]:
    orders = []
    for _ in range(count):
        ts = fake.date_time_between(start_date="-180d", end_date="now")
        orders.append(
            {
                "order_id": f"ORD-{fake.uuid4()[:8].upper()}",
                "timestamp": ts.isoformat(),
                "item_category": random.choice(ECOMMERCE_CATEGORIES),
                "amount": round(random.uniform(199, 25000), 2),
                "merchant_id": f"M-{fake.uuid4()[:6].upper()}",
                "merchant_rating_at_purchase": round(random.uniform(2.5, 5.0), 1),
            }
        )
    return orders


def generate_geo_locations(count: int = 50) -> list[dict]:
    home_lat = random.uniform(INDIA_BOUNDS["lat_min"], INDIA_BOUNDS["lat_max"])
    home_long = random.uniform(INDIA_BOUNDS["long_min"], INDIA_BOUNDS["long_max"])
    work_lat = home_lat + random.uniform(-0.05, 0.05)
    work_long = home_long + random.uniform(-0.05, 0.05)

    locations = []
    base_time = datetime.now() - timedelta(days=30)

    for i in range(count):
        anchor = home_lat if i % 3 != 0 else work_lat
        anchor_long = home_long if i % 3 != 0 else work_long
        locations.append(
            {
                "timestamp": (base_time + timedelta(hours=i * 6)).isoformat(),
                "lat": round(anchor + random.uniform(-0.002, 0.002), 6),
                "long": round(anchor_long + random.uniform(-0.002, 0.002), 6),
                "accuracy_meters": random.randint(5, 50),
            }
        )

    return locations


def generate_cashflow_transactions(count: int = 40) -> list[dict]:
    transactions = []
    base_date = fake.date_between(start_date="-120d", end_date="-1d")

    for i in range(count):
        txn_date = base_date + timedelta(days=i * 3)
        txn_type = random.choice(["CREDIT", "DEBIT", "DEBIT", "DEBIT"])
        ref = fake.uuid4()[:8].upper()
        narration = random.choice(NARRATION_TEMPLATES).format(ref=ref)

        transactions.append(
            {
                "txn_date": txn_date.isoformat(),
                "type": txn_type,
                "amount": round(random.uniform(100, 50000), 2),
                "narration": narration,
            }
        )

    return transactions


def generate_survey() -> dict:
    stress_responses = [
        "I prioritize paying my utility bills first before business inventory purchases.",
        "When money is tight, I delay non-essential spending and focus on rent and groceries.",
        "I sometimes skip savings to buy things I want when I see a sale.",
        "I worry about debt but try to pay creditors on time whenever possible.",
        "I tend to avoid checking my bank balance when I know expenses are high.",
    ]
    return {
        "risk_appetite": random.choice(RISK_APPETITES),
        "savings_freq": random.choice(SAVINGS_FREQS),
        "stress_response_text": random.choice(stress_responses),
    }


def generate_user_profile() -> dict:
    user_id = str(uuid.uuid4())
    return {
        "user_id": user_id,
        "telecom": {"user_id": user_id, "invoices": generate_telecom_invoices()},
        "ecommerce": {"user_id": user_id, "orders": generate_ecommerce_orders()},
        "geo": {"user_id": user_id, "locations": generate_geo_locations()},
        "cashflow": {"user_id": user_id, "transactions": generate_cashflow_transactions()},
        "survey": {"user_id": user_id, **generate_survey()},
    }


def main() -> None:
    profiles = [generate_user_profile() for _ in range(100)]
    OUTPUT_PATH.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    print(f"Generated {len(profiles)} user profiles -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

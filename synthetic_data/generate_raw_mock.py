"""
Generate realistic mock user profiles using a documented generative process.

Generative model (ground truth is hidden from the ML pipeline):
  1. Sample latent creditworthiness theta ~ Beta(2, 2) in [0, 1] (higher = safer).
  2. Sample default outcome Y ~ Bernoulli(p_default) where
     logit(p_default) = intercept + slope * (0.5 - theta) + epsilon.
  3. Observable features are noisy functions of theta (not deterministic rules on features).
  4. Demographic attributes (fairness metadata only, never model inputs) are assigned by
     theta-stratified dealing: each group receives a matched spread of latent
     creditworthiness, so approval-rate parity holds by construction up to sampling
     noise. A small deliberate tilt is applied on geography (urban advantage) so the
     fairness monitor has a realistic, visible—but bounded—disparity to report.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

from preprocessing.clean_geo import latlong_to_pincode

fake = Faker("en_IN")

OUTPUT_PATH = Path(__file__).parent / "mock_data_100_users.json"
GLOBAL_SEED = 42

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
PROTECTED_GROUPS = ["general", "obc", "sc", "st", "minority"]
BORROWER_TYPES = ["individual", "msme"]

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
    "DEBIT/ATM/WDL/{ref}",
    "CREDIT/CASHBACK/{ref}",
    "UPI/BILLPAY/{ref}/Electricity",
]

STRESS_BY_THETA = [
    (
        0.2,
        "I had a sudden medical bill, so I just ignored my utility bills and delayed paying them to cover it.",
        "high",
        "rarely",
    ),
    (
        0.35,
        "When my phone broke, I avoided checking my bank account and just put the repair cost on a credit card that I'll pay late.",
        "high",
        "never",
    ),
    (
        0.55,
        "I had a sudden medical expense; I worried about the debt but managed to pay it off slowly over the month.",
        "medium",
        "monthly",
    ),
    (
        0.75,
        "My laptop needed urgent repairs last month, so I cut back on eating out and used my savings to pay for it immediately.",
        "low",
        "monthly",
    ),
    (
        0.9,
        "I had an unexpected medical bill, but I prioritized paying it immediately from my emergency fund and adjusted my budget.",
        "low",
        "weekly",
    ),
]


def _user_rng(user_id: str) -> random.Random:
    digest = hashlib.sha256(user_id.encode()).hexdigest()
    seed = int(digest[:16], 16)
    return random.Random(seed)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _risk_penalty(theta: float) -> float:
    """Extra bad-signal pressure for the left tail only."""
    return max(0.0, 0.32 - theta) * 2.2


def sample_latent_and_default(rng: random.Random) -> tuple[float, int]:
    """Sample latent creditworthiness and Bernoulli default label.

    Intercept/slope are tuned so calibrated PDs spread across all three decision
    bands (roughly 40/40/20 approve/review/reject on the scorecard) instead of
    piling the whole portfolio into the review band. Tail widening is applied via
    ``_risk_penalty`` on observable features, not by inflating the default prior.

    The latent draw is skewed toward creditworthy (Beta(2.4, 1.9), mean ~0.56):
    an inclusive thin-file lender's applicant pool is mostly people who repay but
    lack a bureau trail, not a random-risk cross-section. The intercept centres the
    portfolio base default rate near ~15% -- low enough that a real share of good
    borrowers clears the approve band, but high enough to leave the champion EBM a
    trainable number of defaulters (a 9% prior starves the model and collapses its
    AUC to ~0.5, which then fails the challenger-agreement gate). The label noise is
    kept tight (sd 0.12) so the theta->default signal is learnable: a champion that
    genuinely discriminates is what lets clear-cut good borrowers earn a *unanimous*
    auto-approval instead of being routed to review on panel disagreement.
    """
    theta = rng.betavariate(2.4, 1.9)
    noise = rng.gauss(0.0, 0.12)
    logit_pd = -1.95 + 4.8 * (0.5 - theta) + noise
    p_default = _sigmoid(logit_pd)
    default_label = 1 if rng.random() < p_default else 0
    return theta, default_label


def generate_telecom_invoices(user_id: str, theta: float, count: int = 12) -> list[dict]:
    rng = _user_rng(f"{user_id}:telecom")
    invoices = []
    base_date = fake.date_between(start_date="-1y", end_date="-1m")
    penalty = _risk_penalty(theta)
    late_prob = max(0.02, 0.35 - 0.30 * theta + 0.28 * penalty + rng.uniform(-0.05, 0.05))
    miss_prob = max(0.01, 0.15 - 0.12 * theta + 0.24 * penalty + rng.uniform(-0.03, 0.03))

    # SIM vintage: high theta -> 24-48 months, low theta -> 3-18 months
    sim_vintage = int(max(1, min(120, round(6 + theta * 36 + rng.uniform(-5, 5)))))

    for i in range(count):
        invoice_date = base_date + timedelta(days=30 * i)
        due_date = invoice_date + timedelta(days=15)
        roll = rng.random()
        if roll < miss_prob:
            status = rng.choice(["defaulted", "missed"])
            payment_date = None
        elif roll < miss_prob + late_prob:
            status = "late"
            payment_date = due_date + timedelta(days=rng.randint(5, 20))
        else:
            status = "paid"
            payment_date = due_date + timedelta(days=rng.randint(-2, 3))

        # Recharge delay (days): high theta -> 0-2 days, low theta -> 1-12 days
        recharge_delay = int(max(0, round((1.0 - theta) * 8 + rng.randint(-2, 4))))

        invoices.append(
            {
                "invoice_date": invoice_date.isoformat(),
                "due_date": due_date.isoformat(),
                "payment_date": payment_date.isoformat() if payment_date else None,
                "billed_amount": round(rng.uniform(299, 1499), 2),
                "status": status,
                "recharge_delay_days": recharge_delay,
                "sim_vintage_months": sim_vintage,
            }
        )
    return invoices


def _pick_delivery_pin(
    rng: random.Random,
    *,
    stable: bool,
    home_pin: str,
    secondary_pin: str,
) -> str:
    if not stable:
        return fake.postcode()
    roll = rng.random()
    if roll < 0.80:
        return home_pin
    if roll < 0.90:
        return secondary_pin
    return fake.postcode()


def generate_ecommerce_orders(user_id: str, theta: float, count: int = 20) -> list[dict]:
    rng = _user_rng(f"{user_id}:ecommerce")
    stable = rng.random() < (0.45 + 0.45 * theta)
    home_lat = rng.uniform(INDIA_BOUNDS["lat_min"], INDIA_BOUNDS["lat_max"])
    home_long = rng.uniform(INDIA_BOUNDS["long_min"], INDIA_BOUNDS["long_max"])
    work_lat = home_lat + rng.uniform(-0.05, 0.05)
    work_long = home_long + rng.uniform(-0.05, 0.05)
    home_pin = latlong_to_pincode(home_lat, home_long)
    secondary_pin = latlong_to_pincode(work_lat, work_long)
    necessity_bias = 0.35 + 0.45 * theta

    orders = []
    for _ in range(count):
        ts = fake.date_time_between(start_date="-180d", end_date="now")
        category = (
            rng.choice(["groceries", "utilities", "health", "education", "household"])
            if rng.random() < necessity_bias
            else rng.choice(ECOMMERCE_CATEGORIES)
        )
        rating_base = 2.8 + 1.8 * theta
        orders.append(
            {
                "order_id": f"ORD-{uuid.uuid4().hex[:8].upper()}",
                "timestamp": ts.isoformat(),
                "item_category": category,
                "amount": round(rng.uniform(199, 25000), 2),
                "merchant_id": f"M-{uuid.uuid4().hex[:6].upper()}",
                "merchant_rating_at_purchase": round(
                    max(2.5, min(5.0, rating_base + rng.uniform(-0.4, 0.4))),
                    1,
                ),
                "delivery_pin_code": _pick_delivery_pin(
                    rng,
                    stable=stable,
                    home_pin=home_pin,
                    secondary_pin=secondary_pin,
                ),
            }
        )
    return orders


def generate_geo_locations(user_id: str, theta: float, count: int = 50) -> list[dict]:
    rng = _user_rng(f"{user_id}:geo")
    home_lat = rng.uniform(INDIA_BOUNDS["lat_min"], INDIA_BOUNDS["lat_max"])
    home_long = rng.uniform(INDIA_BOUNDS["long_min"], INDIA_BOUNDS["long_max"])

    # High theta -> stable shipping anchors (e.g. 1-2 distinct PINs)
    # Low theta -> drifting shipping locations (e.g. 4-8 distinct PINs)
    num_unique_pins = int(max(1, min(10, round(1 + (1.0 - theta) * 7 + rng.uniform(-1, 1)))))

    centroids = []
    for i in range(num_unique_pins):
        spread = 0.05 * i
        centroids.append((home_lat + rng.uniform(-spread, spread), home_long + rng.uniform(-spread, spread)))

    locations = []
    base_time = datetime.now() - timedelta(days=30)
    for i in range(count):
        # Pick centroid: high theta concentrates on index 0, low theta is uniform
        if rng.random() < theta and centroids:
            centroid = centroids[0]
        else:
            centroid = rng.choice(centroids)

        lat, lon = centroid
        locations.append(
            {
                "timestamp": (base_time + timedelta(hours=i * 6)).isoformat(),
                "lat": round(lat + rng.uniform(-0.001, 0.001), 6),
                "long": round(lon + rng.uniform(-0.001, 0.001), 6),
                "accuracy_meters": rng.randint(5, 50),
            }
        )
    return locations


def generate_telecom_sms(user_id: str, theta: float, invoices: list[dict]) -> list[dict]:
    rng = _user_rng(f"{user_id}:telecom_sms")
    sms_records = []
    sender_prefix = rng.choice(["JIO", "BSC"])
    alert_sender = f"AD-{sender_prefix}MOB"
    pay_sender = f"AD-{sender_prefix}PAY"

    for inv in invoices:
        amount = inv["billed_amount"]
        due_dt = datetime.fromisoformat(inv["due_date"])
        # Alert SMS 15 days before due date
        alert_time = due_dt - timedelta(days=15)
        sms_records.append({
            "timestamp": alert_time.isoformat() + "Z",
            "sender": alert_sender,
            "body": f"Dear customer, your bill of Rs {amount} is due on {inv['due_date']}. Please pay promptly."
        })

        # Payment SMS if paid
        pay_str = inv.get("payment_date")
        if pay_str:
            pay_dt = datetime.fromisoformat(pay_str)
            pay_sms_time = pay_dt + timedelta(hours=rng.randint(2, 6))
            sms_records.append({
                "timestamp": pay_sms_time.isoformat() + "Z",
                "sender": pay_sender,
                "body": f"Thank you for your payment of Rs {amount}. Your transaction has been successfully processed."
            })

    sms_records.sort(key=lambda x: x["timestamp"])
    return sms_records


def generate_ecommerce_sms(user_id: str, theta: float, orders: list[dict]) -> list[dict]:
    rng = _user_rng(f"{user_id}:ecommerce_sms")
    sms_records = []

    for order in orders:
        amount = order["amount"]
        ts_str = order["timestamp"]
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if rng.random() < 0.8:
            sms_time = ts + timedelta(minutes=rng.randint(1, 5))
            sms_records.append({
                "timestamp": sms_time.isoformat() + "Z",
                "sender": rng.choice(["AD-BANKTX", "VM-PAYTM", "JD-AMAZON"]),
                "body": f"Alert: Rs {amount} spent on your card at merchant {order['merchant_id']}. Transaction debited."
            })

    sms_records.sort(key=lambda x: x["timestamp"])
    return sms_records


def generate_cashflow_transactions(user_id: str, theta: float, count: int = 40) -> list[dict]:
    rng = _user_rng(f"{user_id}:cashflow")
    base_date = fake.date_between(start_date="-120d", end_date="-1d")
    penalty = _risk_penalty(theta)
    income_scale = max(5000, 10000 + 40000 * theta - 20000 * penalty)
    volatility = max(0.05, 0.35 - 0.25 * theta + 0.22 * penalty)

    transactions = []
    
    # 40% of users have DBT, higher probability for lower theta
    user_has_dbt = rng.random() < (0.6 - 0.4 * theta)

    # Generate 4 monthly cycles (120 days total)
    for m in range(4):
        cycle_start_date = base_date + timedelta(days=m * 30)

        # Day 1: Salary/Income Credit
        salary_ref = uuid.uuid4().hex[:8].upper()
        salary_amount = round(rng.uniform(income_scale * 0.9, income_scale * 1.1), 2)
        transactions.append(
            {
                "txn_date": cycle_start_date.isoformat(),
                "type": "CREDIT",
                "amount": salary_amount,
                "narration": f"NEFT/SALARY/{salary_ref}",
            }
        )

        # Optional DBT Credit
        if user_has_dbt:
            dbt_ref = uuid.uuid4().hex[:8].upper()
            dbt_amount = round(rng.uniform(1500, 3000), 2)
            dbt_date = cycle_start_date + timedelta(days=4)
            transactions.append(
                {
                    "txn_date": dbt_date.isoformat(),
                    "type": "CREDIT",
                    "amount": dbt_amount,
                    "narration": rng.choice([
                        f"DBT/PM-KISAN/{dbt_ref}",
                        f"APBS/PAHAL/{dbt_ref}",
                        f"DBT/SCHOLARSHIP/{dbt_ref}"
                    ])
                }
            )

        # Generate 10 debit transactions per cycle
        for d in range(10):
            if rng.random() > theta:
                # High risk: concentrate debits in the first 7 days
                day_offset = rng.randint(1, 7)
                base_spend = salary_amount * rng.uniform(0.08, 0.15) * (1 + volatility * rng.gauss(0, 0.5))
            else:
                # Low risk: spread debits across the month
                day_offset = rng.randint(2, 29)
                base_spend = salary_amount * rng.uniform(0.04, 0.08) * (1 + volatility * rng.gauss(0, 0.5))

            txn_date = cycle_start_date + timedelta(days=day_offset)
            ref = uuid.uuid4().hex[:8].upper()
            
            # 20% of transactions are UPI-LITE
            is_upi_lite = rng.random() < 0.2
            if is_upi_lite:
                amount = round(rng.uniform(10, 200), 2)
                narration = rng.choice([
                    f"UPI-LITE/MERCHANT/{ref}/Tea",
                    f"UPI/LITE-WALLET/{ref}/Auto",
                    f"LITE-WALLET/TRANSFER/{ref}/Snacks"
                ])
            else:
                amount = round(max(50.0, base_spend), 2)
                narration = rng.choice(NARRATION_TEMPLATES).format(ref=ref)

            transactions.append(
                {
                    "txn_date": txn_date.isoformat(),
                    "type": "DEBIT",
                    "amount": amount,
                    "narration": narration,
                }
            )

    transactions.sort(key=lambda x: x["txn_date"])
    return transactions


# ---------------------------------------------------------------------------
# Cohort-specific alternative data sources (Campus / Vendor / Farmer / Household)
# ---------------------------------------------------------------------------
# These cohorts are scored on a dedicated facet whose signals live in a source the
# generic streams (telecom / e-commerce / cash-flow) don't carry -- a student's UPI
# canteen spend, a vendor's QR settlements, a farmer's mandi receipts, a homemaker's
# utility bills. The builders below reconstruct a transaction-level raw payload from
# the borrower's already-derived facet feature values, so the Score Explainer's Step
# 2 (raw vault) reconciles exactly with the Step 3 features and the scorecard. They
# are deterministic per user id, so the same borrower always shows the same records.


def build_campus_raw_payload(
    user_id: str,
    *,
    upi_spend_consistency: float,
    small_dues_payment_promptness: float,
    e_wallet_topup_frequency: float,
) -> dict:
    """Student campus & UPI footprint: canteen/stationery spend, mess dues, wallet top-ups."""
    rng = _user_rng(f"{user_id}:campus_display")
    start = datetime.now() - timedelta(days=45)

    # Consistent spenders show a tight amount distribution (low coefficient of variation).
    cv = max(0.03, 0.5 * (1.0 - upi_spend_consistency))
    base_amount = rng.uniform(60, 180)
    merchants = ["Campus Canteen", "Stationery Store", "Xerox Point", "Book Depot", "Chai Tapri", "Metro Card"]
    upi_transactions = [
        {
            "timestamp": (start + timedelta(days=i * 3, hours=rng.randint(8, 21))).isoformat(),
            "merchant": rng.choice(merchants),
            "amount": round(max(15.0, base_amount * (1 + rng.gauss(0, cv))), 2),
            "channel": "UPI",
        }
        for i in range(14)
    ]

    # Promptness is the share of small dues settled on or before the due date.
    small_dues_ledger = []
    for i in range(6):
        due = start + timedelta(days=i * 7)
        on_time = rng.random() < small_dues_payment_promptness
        settled = due + timedelta(days=(rng.randint(-1, 1) if on_time else rng.randint(4, 15)))
        small_dues_ledger.append(
            {
                "description": rng.choice(["Hostel mess share", "Group project print", "Trip split", "Notes subscription"]),
                "amount": round(rng.uniform(150, 900), 2),
                "due_date": due.date().isoformat(),
                "settled_date": settled.date().isoformat(),
                "on_time": on_time,
            }
        )

    # Top-up frequency (0..1) maps to a monthly count of e-wallet reloads.
    topup_count = int(round(2 + e_wallet_topup_frequency * 10))
    wallet_topups = [
        {
            "timestamp": (start + timedelta(days=rng.randint(0, 44))).isoformat(),
            "wallet": rng.choice(["Paytm", "PhonePe", "Amazon Pay"]),
            "amount": round(rng.choice([100, 200, 500]) + rng.uniform(0, 50), 2),
        }
        for _ in range(topup_count)
    ]

    return {
        "user_id": user_id,
        "source": "Campus & UPI transaction behaviour",
        "upi_transactions": upi_transactions,
        "small_dues_ledger": small_dues_ledger,
        "wallet_topups": wallet_topups,
        "derived_metrics": {
            "upi_spend_consistency": round(upi_spend_consistency, 2),
            "small_dues_payment_promptness": round(small_dues_payment_promptness, 2),
            "e_wallet_topup_frequency": round(e_wallet_topup_frequency, 2),
        },
    }


def build_vendor_raw_payload(
    user_id: str,
    *,
    daily_transaction_count: float,
    average_ticket_size: float,
) -> dict:
    """Micro-enterprise UPI/QR settlements: daily volume and average ticket size."""
    rng = _user_rng(f"{user_id}:vendor_display")
    start = datetime.now() - timedelta(days=14)

    daily_settlements = []
    for d in range(14):
        day = start + timedelta(days=d)
        count = max(1, int(round(daily_transaction_count * (1 + rng.gauss(0, 0.18)))))
        avg = max(20.0, average_ticket_size * (1 + rng.gauss(0, 0.12)))
        daily_settlements.append(
            {
                "date": day.date().isoformat(),
                "txn_count": count,
                "gross_volume": round(count * avg, 2),
                "avg_ticket": round(avg, 2),
                "provider": rng.choice(["BharatPe QR", "Paytm Soundbox", "PhonePe Merchant"]),
            }
        )

    return {
        "user_id": user_id,
        "source": "Micro-enterprise UPI/payment volumes",
        "daily_settlements": daily_settlements,
        "derived_metrics": {
            "daily_transaction_count": round(daily_transaction_count, 2),
            "average_ticket_size": round(average_ticket_size, 2),
        },
    }


def build_farmer_raw_payload(
    user_id: str,
    *,
    harvest_income_spike: float,
    input_purchase_consistency: float,
    enam_receipt_volume: float,
) -> dict:
    """Farming cycle footprint: e-NAM mandi receipts and seasonal input purchases."""
    rng = _user_rng(f"{user_id}:farmer_display")
    start = datetime.now() - timedelta(days=300)

    # Distribute the annual e-NAM verified volume across a handful of mandi sales,
    # concentrating it into a harvest-season spike proportional to the income multiple.
    n_receipts = 5
    weights = [rng.uniform(0.5, 1.0) for _ in range(n_receipts)]
    peak = max(1.0, harvest_income_spike)
    weights[rng.randrange(n_receipts)] *= peak  # the harvest-season spike
    wsum = sum(weights) or 1.0
    enam_receipts = []
    for i, w in enumerate(weights):
        enam_receipts.append(
            {
                "receipt_date": (start + timedelta(days=i * 55 + rng.randint(0, 20))).date().isoformat(),
                "mandi": rng.choice(["Azadpur APMC", "Unjha Mandi", "Nashik APMC", "Guntur Yard"]),
                "commodity": rng.choice(["Paddy", "Wheat", "Cotton", "Turmeric"]),
                "amount": round(enam_receipt_volume * w / wsum, 2),
                "verified": True,
            }
        )

    # Consistency is the share of input-purchase cycles the farmer actually stocked.
    input_purchases = []
    for i in range(4):
        cycle = start + timedelta(days=i * 75)
        stocked = rng.random() < input_purchase_consistency
        if stocked:
            input_purchases.append(
                {
                    "purchase_date": (cycle + timedelta(days=rng.randint(0, 10))).date().isoformat(),
                    "item": rng.choice(["Certified seeds", "Urea/DAP fertiliser", "Pesticide", "Diesel"]),
                    "amount": round(rng.uniform(3000, 12000), 2),
                    "channel": rng.choice(["Kisan Credit Card", "Co-op society", "Agri-input dealer"]),
                }
            )

    return {
        "user_id": user_id,
        "source": "Farming cycles & input purchases",
        "enam_receipts": enam_receipts,
        "input_purchases": input_purchases,
        "derived_metrics": {
            "harvest_income_spike": round(harvest_income_spike, 2),
            "input_purchase_consistency": round(input_purchase_consistency, 2),
            "enam_receipt_volume": round(enam_receipt_volume, 2),
        },
    }


def build_household_raw_payload(
    user_id: str,
    *,
    utility_payment_consistency: float,
    grocery_spend_stability: float,
) -> dict:
    """Homemaker footprint: BBPS utility bills and recurring grocery spend."""
    rng = _user_rng(f"{user_id}:household_display")
    start = datetime.now() - timedelta(days=180)

    # Consistency is the share of utility bills paid on or before the due date.
    utility_bills = []
    for i in range(6):
        biller = ["Electricity", "Water", "Piped Gas"][i % 3]
        due = start + timedelta(days=i * 30)
        on_time = rng.random() < utility_payment_consistency
        paid = due + timedelta(days=(rng.randint(-2, 0) if on_time else rng.randint(3, 12)))
        utility_bills.append(
            {
                "biller": biller,
                "channel": "BBPS",
                "amount": round(rng.uniform(300, 1600), 2),
                "due_date": due.date().isoformat(),
                "paid_date": paid.date().isoformat(),
                "on_time": on_time,
            }
        )

    # Stability shows up as low month-to-month variance in grocery spend.
    cv = max(0.02, 0.4 * (1.0 - grocery_spend_stability))
    base_spend = rng.uniform(4000, 9000)
    grocery_spend = [
        {
            "month": (start + timedelta(days=i * 30)).strftime("%Y-%m"),
            "amount": round(max(500.0, base_spend * (1 + rng.gauss(0, cv))), 2),
            "outlet": rng.choice(["Local Kirana", "D-Mart", "Reliance Fresh", "Grocery POS"]),
        }
        for i in range(6)
    ]

    return {
        "user_id": user_id,
        "source": "Electricity/Water/Gas & Groceries",
        "utility_bills": utility_bills,
        "grocery_spend": grocery_spend,
        "derived_metrics": {
            "utility_payment_consistency": round(utility_payment_consistency, 2),
            "grocery_spend_stability": round(grocery_spend_stability, 2),
        },
    }


# Maps each cohort to its dedicated data source: (vault data_type, builder, feature
# names in the builder's kwarg order, per-feature fallback when the borrower has no
# stored value). Used by the Score Explainer to materialise the raw vault on demand.
COHORT_RAW_SOURCE_BUILDERS = {
    "Student": (
        "campus",
        build_campus_raw_payload,
        (
            ("upi_spend_consistency", 0.65),
            ("small_dues_payment_promptness", 0.74),
            ("e_wallet_topup_frequency", 0.59),
        ),
    ),
    "Vendor": (
        "vendor",
        build_vendor_raw_payload,
        (
            ("daily_transaction_count", 31.0),
            ("average_ticket_size", 176.0),
        ),
    ),
    "Farmer": (
        "farmer",
        build_farmer_raw_payload,
        (
            ("harvest_income_spike", 5.3),
            ("input_purchase_consistency", 0.85),
            ("enam_receipt_volume", 120000.0),
        ),
    ),
    "Homemaker": (
        "household",
        build_household_raw_payload,
        (
            ("utility_payment_consistency", 0.9),
            ("grocery_spend_stability", 0.8),
        ),
    ),
}


def _noisy_trait(theta: float, rng: random.Random, noise: float = 0.12) -> float:
    value = theta + rng.gauss(0.0, noise)
    return round(max(0.0, min(1.0, value)), 4)


def generate_survey(user_id: str, theta: float, extra_features: dict | None = None) -> dict:
    """Generate psychometric trait payload as noisy functions of latent creditworthiness."""
    rng = _user_rng(f"{user_id}:psychometric")
    traits = {
        "conscientiousness": _noisy_trait(theta, rng),
        "locus_of_control": _noisy_trait(theta, rng),
        "financial_self_efficacy": _noisy_trait(theta, rng),
        "present_bias": _noisy_trait(1.0 - theta, rng),
        "debt_attitude": _noisy_trait(theta, rng),
        "risk_tolerance": _noisy_trait(1.0 - theta, rng),
        "delayed_gratification": _noisy_trait(theta, rng),
        "honesty": _noisy_trait(theta, rng),
        "cognitive_reflection": _noisy_trait(theta, rng),
        "resourcefulness": _noisy_trait(theta, rng),
        "response_validity": round(max(0.6, min(1.0, 0.85 + rng.gauss(0, 0.08))), 4),
    }
    if extra_features:
        traits.update(extra_features)
    best = min(STRESS_BY_THETA, key=lambda item: abs(item[0] - theta))
    _, text, _, _ = best
    return {
        "language": "en",
        "assessment_version": "1.0",
        "traits": traits,
        **traits,
        "answers": {"open_1": text},
        "transcript": [{"role": "user", "text": text}],
    }


# Fairness-metadata group shares (counts per 100 borrowers).
DEMOGRAPHIC_QUOTAS: dict[str, dict[str, int]] = {
    "gender": {"male": 52, "female": 45, "other": 3},
    "geography": {"rural": 40, "semi_urban": 35, "urban": 25},
    "protected_group": {"general": 20, "obc": 20, "sc": 20, "st": 20, "minority": 20},
    "borrower_type": {"individual": 75, "msme": 25},
}

# Deliberate mild disparity: (attribute, favored group, disfavored group, #swaps).
# A few rank swaps nudge urban borrowers toward higher theta, leaving a visible but
# bounded approval-rate gap for the fairness monitor to surface.
DEMOGRAPHIC_TILTS: list[tuple[str, str, str, int]] = [
    ("geography", "urban", "rural", 3),
]


def _deal_balanced(order: list[int], quotas: dict[str, int]) -> dict[int, str]:
    """Deal group labels over theta-ranked users with a weighted round-robin.

    Each group's members end up evenly spaced along the theta ranking, so every
    group sees a matched distribution of latent creditworthiness.
    """
    n = len(order)
    total = sum(quotas.values())
    allocated = {group: 0 for group in quotas}
    labels: dict[int, str] = {}
    for pos, idx in enumerate(order):
        # Pick the group furthest behind its pro-rata share (largest remainder).
        group = max(quotas, key=lambda g: quotas[g] * (pos + 1) / total - allocated[g])
        labels[idx] = group
        allocated[group] += 1
    return labels


def _apply_tilt(
    labels: dict[int, str], order: list[int], favored: str, disfavored: str, swaps: int
) -> None:
    """Swap ``swaps`` label pairs so ``favored`` drifts toward higher theta ranks."""
    favored_positions = [p for p, idx in enumerate(order) if labels[idx] == favored]
    disfavored_positions = [p for p, idx in enumerate(order) if labels[idx] == disfavored]
    done = 0
    for low in favored_positions:  # lowest-ranked favored members first
        candidates = [p for p in disfavored_positions if p > low]
        if not candidates:
            break
        high = candidates[-1]  # highest-ranked disfavored member
        labels[order[low]], labels[order[high]] = labels[order[high]], labels[order[low]]
        disfavored_positions.remove(high)
        done += 1
        if done >= swaps:
            break


def assign_demographics(thetas: list[float]) -> list[dict[str, str]]:
    """Assign fairness metadata via theta-stratified dealing (see module docstring)."""
    order = sorted(range(len(thetas)), key=lambda i: thetas[i])
    per_attribute: dict[str, dict[int, str]] = {}
    for attribute, quotas in DEMOGRAPHIC_QUOTAS.items():
        labels = _deal_balanced(order, quotas)
        per_attribute[attribute] = labels
    for attribute, favored, disfavored, swaps in DEMOGRAPHIC_TILTS:
        _apply_tilt(per_attribute[attribute], order, favored, disfavored, swaps)
    return [
        {attribute: per_attribute[attribute][i] for attribute in DEMOGRAPHIC_QUOTAS}
        for i in range(len(thetas))
    ]


COHORT_SHARING_PROB = {
    "Salaried": {"telecom": 0.95, "ecommerce": 0.90, "geo": 0.95, "cashflow": 0.95},
    "GigWorker": {"telecom": 0.90, "ecommerce": 0.85, "geo": 0.90, "cashflow": 0.80},
    "Student": {"telecom": 0.40, "ecommerce": 0.50, "geo": 0.95, "cashflow": 0.20},
    "Vendor": {"telecom": 0.70, "ecommerce": 0.40, "geo": 0.90, "cashflow": 0.75},
    "Farmer": {"telecom": 0.50, "ecommerce": 0.15, "geo": 0.90, "cashflow": 0.40},
    "Homemaker": {"telecom": 0.85, "ecommerce": 0.40, "geo": 0.85, "cashflow": 0.30},
}


def generate_user_profile(
    theta: float,
    default_label: int,
    demographics: dict[str, str],
    rng: random.Random,
) -> dict:
    # Seeded UUID: the whole cohort (ids, and therefore every per-user data stream,
    # which is keyed on the id) is reproducible from GLOBAL_SEED.
    user_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
    local_rng = rng
    protected_group = demographics["protected_group"]
    borrower_type = demographics["borrower_type"]

    cohort = local_rng.choice(["Salaried", "GigWorker", "Student", "Vendor", "Farmer", "Homemaker"])
    cohort_codes = {
        "Salaried": 0.0,
        "GigWorker": 1.0,
        "Student": 2.0,
        "Vendor": 3.0,
        "Farmer": 4.0,
        "Homemaker": 5.0,
    }
    cohort_code = cohort_codes[cohort]

    # Demographic dimensions, used only for fairness monitoring, never as model inputs.
    gender = demographics["gender"]
    geography = demographics["geography"]

    extra_features = {
        "cohort_code": cohort_code,
    }

    if cohort == "Student":
        extra_features["upi_spend_consistency"] = round(local_rng.uniform(0.5, 1.0) if theta > 0.4 else local_rng.uniform(0.2, 0.65), 2)
        extra_features["small_dues_payment_promptness"] = round(local_rng.uniform(0.6, 1.0) if theta > 0.4 else local_rng.uniform(0.3, 0.75), 2)
        extra_features["e_wallet_topup_frequency"] = round(local_rng.uniform(0.5, 1.0) if theta > 0.5 else local_rng.uniform(0.0, 0.6), 2)
    elif cohort == "Vendor":
        extra_features["daily_transaction_count"] = round(local_rng.uniform(15, 60) if theta > 0.3 else local_rng.uniform(5, 25), 2)
        extra_features["average_ticket_size"] = round(local_rng.uniform(50, 500) if theta > 0.4 else local_rng.uniform(20, 150), 2)
    elif cohort == "Farmer":
        extra_features["harvest_income_spike"] = round(local_rng.uniform(3.0, 10.0) if theta > 0.4 else local_rng.uniform(1.0, 4.0), 2)
        extra_features["input_purchase_consistency"] = round(local_rng.uniform(0.7, 1.0) if theta > 0.4 else local_rng.uniform(0.3, 0.75), 2)
        extra_features["enam_receipt_volume"] = round(local_rng.uniform(50000, 250000) if theta > 0.4 else local_rng.uniform(10000, 75000), 2)
    elif cohort == "Homemaker":
        extra_features["utility_payment_consistency"] = round(local_rng.uniform(0.8, 1.0) if theta > 0.4 else local_rng.uniform(0.4, 0.85), 2)
        extra_features["grocery_spend_stability"] = round(local_rng.uniform(0.7, 1.0) if theta > 0.4 else local_rng.uniform(0.3, 0.75), 2)

    # Business cohorts also declare a business profile at onboarding: how long the
    # business has run, and how well their self-reported turnover agrees with the
    # observed cash-flow (the consistency, not the claim, tracks creditworthiness).
    if cohort in ("Vendor", "Farmer"):
        extra_features["business_vintage_years"] = float(
            max(1, min(15, round(1 + theta * 11 + local_rng.uniform(-1.5, 1.5))))
        )
        extra_features["turnover_income_consistency"] = round(
            max(0.0, min(1.0, 0.45 + 0.5 * theta + local_rng.gauss(0.0, 0.08))), 4
        )

    profile = {
        "user_id": user_id,
        "survey": {"user_id": user_id, **generate_survey(user_id, theta, extra_features)},
        "_ground_truth": {
            "latent_creditworthiness": round(theta, 4),
            "default_label": default_label,
            "protected_group": protected_group,
            "borrower_type": borrower_type,
            "cohort": cohort,
            "gender": gender,
            "geography": geography,
        },
    }

    sharing_probs = COHORT_SHARING_PROB.get(cohort, {})
    if local_rng.random() < sharing_probs.get("telecom", 1.0):
        invoices = generate_telecom_invoices(user_id, theta)
        sms_records = generate_telecom_sms(user_id, theta, invoices)
        profile["telecom"] = {"user_id": user_id, "invoices": invoices, "sms_records": sms_records}
    if local_rng.random() < sharing_probs.get("ecommerce", 1.0):
        orders = generate_ecommerce_orders(user_id, theta)
        sms_records = generate_ecommerce_sms(user_id, theta, orders)
        profile["ecommerce"] = {"user_id": user_id, "orders": orders, "sms_records": sms_records}
    if local_rng.random() < sharing_probs.get("geo", 1.0):
        profile["geo"] = {"user_id": user_id, "locations": generate_geo_locations(user_id, theta)}
    if local_rng.random() < sharing_probs.get("cashflow", 1.0):
        profile["cashflow"] = {"user_id": user_id, "transactions": generate_cashflow_transactions(user_id, theta)}

    if borrower_type == "msme":
        profile["msme"] = {
            "user_id": user_id,
            "business_name": fake.company(),
            "gst_turnover_monthly": round(50000 + 450000 * theta + local_rng.uniform(-20000, 20000), 2),
            "merchant_rating_avg": round(2.8 + 1.8 * theta + local_rng.uniform(-0.3, 0.3), 2),
            "years_in_business": local_rng.randint(1, 12),
        }
    return profile


def main() -> None:
    master_rng = random.Random(GLOBAL_SEED)
    Faker.seed(GLOBAL_SEED)  # dates/addresses drawn via faker are reproducible too
    count = 100
    latents = [sample_latent_and_default(master_rng) for _ in range(count)]
    demographics = assign_demographics([theta for theta, _ in latents])
    profiles = [
        generate_user_profile(theta, default_label, demographics[i], master_rng)
        for i, (theta, default_label) in enumerate(latents)
    ]
    OUTPUT_PATH.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    default_rate = sum(p["_ground_truth"]["default_label"] for p in profiles) / len(profiles)
    print(f"Generated {len(profiles)} user profiles -> {OUTPUT_PATH}")
    print(f"  Synthetic default rate: {default_rate:.1%}")
    print(f"  MSME profiles: {sum(1 for p in profiles if p['_ground_truth']['borrower_type'] == 'msme')}")


if __name__ == "__main__":
    main()

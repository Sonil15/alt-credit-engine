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
    piling the whole portfolio into the approve band. Tail widening is applied via
    ``_risk_penalty`` on observable features, not by inflating the default prior.
    """
    theta = rng.betavariate(2.0, 2.0)
    noise = rng.gauss(0.0, 0.25)
    logit_pd = -1.55 + 4.0 * (0.5 - theta) + noise
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

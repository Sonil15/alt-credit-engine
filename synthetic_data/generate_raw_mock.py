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
        "I sometimes skip savings to buy things I want when I see a sale.",
        "high",
        "rarely",
    ),
    (
        0.35,
        "I tend to avoid checking my bank balance when I know expenses are high.",
        "high",
        "never",
    ),
    (
        0.55,
        "I worry about debt but try to pay creditors on time whenever possible.",
        "medium",
        "monthly",
    ),
    (
        0.75,
        "When money is tight, I delay non-essential spending and focus on rent and groceries.",
        "low",
        "monthly",
    ),
    (
        0.9,
        "I prioritize paying my utility bills first before business inventory purchases.",
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


def sample_latent_and_default(rng: random.Random) -> tuple[float, int]:
    """Sample latent creditworthiness and Bernoulli default label.

    Intercept/slope are tuned so calibrated PDs spread across all three decision
    bands (roughly 40/40/20 approve/review/reject on the scorecard) instead of
    piling the whole portfolio into the approve band.
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
    late_prob = max(0.02, 0.35 - 0.30 * theta + rng.uniform(-0.05, 0.05))
    miss_prob = max(0.01, 0.15 - 0.12 * theta + rng.uniform(-0.03, 0.03))

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

        invoices.append(
            {
                "invoice_date": invoice_date.isoformat(),
                "due_date": due_date.isoformat(),
                "payment_date": payment_date.isoformat() if payment_date else None,
                "billed_amount": round(rng.uniform(299, 1499), 2),
                "status": status,
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
    spread = max(0.002, 0.02 - 0.018 * theta)
    work_lat = home_lat + rng.uniform(-spread, spread)
    work_long = home_long + rng.uniform(-spread, spread)

    locations = []
    base_time = datetime.now() - timedelta(days=30)
    for i in range(count):
        anchor_lat = home_lat if i % 3 != 0 else work_lat
        anchor_long = home_long if i % 3 != 0 else work_long
        locations.append(
            {
                "timestamp": (base_time + timedelta(hours=i * 6)).isoformat(),
                "lat": round(anchor_lat + rng.uniform(-spread, spread), 6),
                "long": round(anchor_long + rng.uniform(-spread, spread), 6),
                "accuracy_meters": rng.randint(5, 50),
            }
        )
    return locations


def generate_cashflow_transactions(user_id: str, theta: float, count: int = 40) -> list[dict]:
    rng = _user_rng(f"{user_id}:cashflow")
    base_date = fake.date_between(start_date="-120d", end_date="-1d")
    income_scale = 8000 + 42000 * theta
    volatility = max(0.05, 0.35 - 0.25 * theta)

    transactions = []
    for i in range(count):
        txn_date = base_date + timedelta(days=i * 3)
        is_income = rng.random() < (0.18 + 0.12 * theta)
        txn_type = "CREDIT" if is_income else "DEBIT"
        ref = uuid.uuid4().hex[:8].upper()
        if is_income:
            amount = round(rng.uniform(income_scale * 0.8, income_scale * 1.2), 2)
            narration = f"NEFT/SALARY/{ref}"
        else:
            base_spend = income_scale * rng.uniform(0.15, 0.45) * (1 + volatility * rng.gauss(0, 1))
            amount = round(max(100, base_spend), 2)
            narration = rng.choice(NARRATION_TEMPLATES).format(ref=ref)

        transactions.append(
            {
                "txn_date": txn_date.isoformat(),
                "type": txn_type,
                "amount": amount,
                "narration": narration,
            }
        )
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
        "telecom": {"user_id": user_id, "invoices": generate_telecom_invoices(user_id, theta)},
        "ecommerce": {"user_id": user_id, "orders": generate_ecommerce_orders(user_id, theta)},
        "geo": {"user_id": user_id, "locations": generate_geo_locations(user_id, theta)},
        "cashflow": {"user_id": user_id, "transactions": generate_cashflow_transactions(user_id, theta)},
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

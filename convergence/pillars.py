"""Per-pillar sub-scores mapping directly to the problem-statement data sources.

Each borrower is summarised across five interpretable pillars (0-100), one per
alternative-data source the bank ingests. Scores are normalised *relative to the
scored population* (winsorised p10-p90) rather than against hand-tuned magic
constants, so they adapt to whatever scale the cleaners produce and stay
meaningful as the data distribution shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from core.json_utils import safe_float

# direction: "high" = larger raw value is better, "low" = larger is worse.
PillarFeature = tuple[str, str, float]  # (feature_name, direction, weight)


@dataclass(frozen=True)
class Pillar:
    key: str
    label: str
    source: str  # plain-language data source for the UI
    features: tuple[PillarFeature, ...]


class BorrowerCohort(str, Enum):
    SALARIED = "Salaried"
    GIG_WORKER = "GigWorker"
    STUDENT = "Student"
    VENDOR = "Vendor"
    FARMER = "Farmer"
    HOMEMAKER = "Homemaker"


COHORT_CODE_MAP = {
    0.0: BorrowerCohort.SALARIED,
    1.0: BorrowerCohort.GIG_WORKER,
    2.0: BorrowerCohort.STUDENT,
    3.0: BorrowerCohort.VENDOR,
    4.0: BorrowerCohort.FARMER,
    5.0: BorrowerCohort.HOMEMAKER,
}


COHORT_EXPECTED_PILLARS: dict[BorrowerCohort, list[str]] = {
    BorrowerCohort.SALARIED: [
        "telecom_reliability",
        "spending_behaviour",
        "location_stability",
        "cashflow_resilience",
        "psychometric_character",
    ],
    BorrowerCohort.GIG_WORKER: [
        "telecom_reliability",
        "spending_behaviour",
        "location_stability",
        "cashflow_resilience",
        "psychometric_character",
    ],
    BorrowerCohort.STUDENT: [
        "location_stability",
        "psychometric_character",
        "campus_transaction_behavior",
    ],
    BorrowerCohort.VENDOR: [
        "location_stability",
        "psychometric_character",
        "vendor_transaction_velocity",
    ],
    BorrowerCohort.FARMER: [
        "location_stability",
        "psychometric_character",
        "agricultural_seasonality",
    ],
    BorrowerCohort.HOMEMAKER: [
        "telecom_reliability",
        "location_stability",
        "psychometric_character",
        "household_reliability",
    ],
}


PILLARS: tuple[Pillar, ...] = (
    Pillar(
        key="telecom_reliability",
        label="Telecom Reliability",
        source="Mobile & broadband payments",
        features=(
            ("avg_days_late", "low", 0.4),
            ("missed_payments_count", "low", 0.6),
        ),
    ),
    Pillar(
        key="spending_behaviour",
        label="Spending Behaviour",
        source="E-commerce purchase patterns",
        features=(
            ("necessity_ratio", "high", 0.4),
            ("avg_merchant_rating", "high", 0.3),
            ("monthly_spend_volatility", "low", 0.3),
        ),
    ),
    Pillar(
        key="location_stability",
        label="Location Stability",
        source="Geolocation & delivery consistency",
        features=(
            ("spatial_variance_score", "low", 0.6),
            ("anchor_count", "high", 0.4),
        ),
    ),
    Pillar(
        key="cashflow_resilience",
        label="Cashflow Resilience",
        source="Bank cash-flow (econometric ECM)",
        features=(
            ("monthly_income_mean", "high", 0.25),
            ("cashflow_volatility", "low", 0.25),
            ("resilience_coefficient", "high", 0.25),
            ("trend_slope", "high", 0.20),
            ("is_stationary", "high", 0.05),
        ),
    ),
    Pillar(
        key="psychometric_character",
        label="Psychometric Character",
        source="Behavioural assessment",
        features=(
            ("conscientiousness", "high", 0.1),
            ("locus_of_control", "high", 0.1),
            ("financial_self_efficacy", "high", 0.1),
            ("present_bias", "low", 0.1),
            ("debt_attitude", "high", 0.1),
            ("risk_tolerance", "low", 0.1),
            ("delayed_gratification", "high", 0.1),
            ("honesty", "high", 0.1),
            ("cognitive_reflection", "high", 0.1),
            ("resourcefulness", "high", 0.1),
        ),
    ),
    Pillar(
        key="campus_transaction_behavior",
        label="Campus & UPI Transaction Behavior",
        source="UPI expenses & small dues history",
        features=(
            ("upi_spend_consistency", "high", 0.4),
            ("small_dues_payment_promptness", "high", 0.4),
            ("e_wallet_topup_frequency", "high", 0.2),
        ),
    ),
    Pillar(
        key="vendor_transaction_velocity",
        label="Vendor Transaction Velocity",
        source="Micro-enterprise UPI/payment volumes",
        features=(
            ("daily_transaction_count", "high", 0.5),
            ("average_ticket_size", "high", 0.5),
        ),
    ),
    Pillar(
        key="agricultural_seasonality",
        label="Agricultural Seasonality",
        source="Farming cycles & input purchases",
        features=(
            ("harvest_income_spike", "high", 0.6),
            ("input_purchase_consistency", "high", 0.4),
        ),
    ),
    Pillar(
        key="household_reliability",
        label="Household Reliability",
        source="Electricity/Water/Gas & Groceries",
        features=(
            ("utility_payment_consistency", "high", 0.6),
            ("grocery_spend_stability", "high", 0.4),
        ),
    ),
)

ALL_PILLAR_FEATURES = [feat for pillar in PILLARS for feat, _, _ in pillar.features]


def compute_norm_stats(wide: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Winsorised p10/p90 bounds per feature, computed across the population."""
    stats: dict[str, tuple[float, float]] = {}
    for feature in ALL_PILLAR_FEATURES:
        if feature in wide.columns:
            col = pd.to_numeric(wide[feature], errors="coerce").dropna()
        else:
            col = pd.Series(dtype=float)
        if len(col) >= 2:
            lo = float(np.percentile(col, 10))
            hi = float(np.percentile(col, 90))
        elif len(col) == 1:
            lo = hi = float(col.iloc[0])
        else:
            lo = hi = 0.0
        stats[feature] = (lo, hi)
    return stats


def feature_goodness(value: float, lo: float, hi: float, direction: str) -> float:
    """Map a raw feature value to a 0-1 'goodness' score given population bounds."""
    value = safe_float(value)
    if hi - lo < 1e-9:
        return 0.5  # no spread in the population -> neutral
    fraction = (value - lo) / (hi - lo)
    fraction = max(0.0, min(1.0, fraction))
    return fraction if direction == "high" else 1.0 - fraction


def _pillar_has_data(row: pd.Series, features: tuple[PillarFeature, ...]) -> bool:
    for feat, _, _ in features:
        if feat in row and abs(safe_float(row.get(feat, 0.0))) > 1e-9:
            return True
    return False


def grade(score: float) -> str:
    if score >= 75:
        return "Strong"
    if score >= 55:
        return "Adequate"
    if score >= 35:
        return "Weak"
    return "Poor"


def compute_pillar_scores(
    row: pd.Series,
    norm_stats: dict[str, tuple[float, float]],
    cohort: BorrowerCohort | str | None = None,
) -> list[dict]:
    """Return a list of pillar dicts (score 0-100, grade, data presence)."""
    if cohort is None:
        if "cohort_code" in row:
            code = safe_float(row.get("cohort_code", 0.0))
            cohort = COHORT_CODE_MAP.get(code, BorrowerCohort.SALARIED)
        else:
            cohort_str = row.get("cohort", "Salaried")
            if isinstance(cohort_str, float) or pd.isna(cohort_str) or cohort_str is None:
                cohort = BorrowerCohort.SALARIED
            else:
                try:
                    cohort = BorrowerCohort(str(cohort_str))
                except ValueError:
                    cohort = BorrowerCohort.SALARIED
    elif isinstance(cohort, str):
        try:
            cohort = BorrowerCohort(cohort)
        except ValueError:
            cohort = BorrowerCohort.SALARIED

    expected_keys = COHORT_EXPECTED_PILLARS.get(cohort, COHORT_EXPECTED_PILLARS[BorrowerCohort.SALARIED])
    filtered_pillars = [
        p for p in PILLARS
        if p.key in expected_keys or _pillar_has_data(row, p.features)
    ]

    results: list[dict] = []
    for pillar in filtered_pillars:
        weighted = 0.0
        total_weight = 0.0
        contributing: list[dict] = []
        for feat, direction, weight in pillar.features:
            # Dynamic weights adjustment for Gig Worker
            if cohort == BorrowerCohort.GIG_WORKER:
                if feat == "cashflow_volatility":
                    weight = 0.05
                elif feat == "monthly_income_mean":
                    weight = 0.35
                elif feat == "resilience_coefficient":
                    weight = 0.35
                elif feat == "missed_payments_count":
                    weight = 0.2
                elif feat == "avg_days_late":
                    weight = 0.8

            feat_val = row.get(feat)
            if pd.isna(feat_val):
                contributing.append(
                    {
                        "feature": feat,
                        "value": None,
                        "goodness": None,
                    }
                )
                continue

            lo, hi = norm_stats.get(feat, (0.0, 0.0))
            goodness = feature_goodness(feat_val, lo, hi, direction)
            weighted += goodness * weight
            total_weight += weight
            contributing.append(
                {
                    "feature": feat,
                    "value": safe_float(feat_val),
                    "goodness": round(goodness, 3),
                }
            )
        score = round(100.0 * weighted / total_weight, 1) if total_weight else 0.0
        results.append(
            {
                "key": pillar.key,
                "label": pillar.label,
                "source": pillar.source,
                "score": score,
                "grade": grade(score),
                "has_data": _pillar_has_data(row, pillar.features),
                "features": contributing,
            }
        )
    return results


def compute_confidence(
    pillar_scores: list[dict],
    cohort: BorrowerCohort | str | None = None,
) -> dict:
    """Data-sufficiency / confidence from how many features are backed by real data."""
    total_pillars = len(pillar_scores) or 1
    pillars_with_data = sum(1 for p in pillar_scores if p["has_data"])
    
    total_features = 0
    features_with_data = 0
    for p in pillar_scores:
        total_features += len(p["features"])
        if p["has_data"] and p["key"] != "psychometric_character":
            features_with_data += len(p["features"])
        else:
            for f in p["features"]:
                if f["value"] is not None and abs(safe_float(f["value"])) > 1e-9:
                    features_with_data += 1

    if cohort is not None:
        if isinstance(cohort, str):
            try:
                cohort = BorrowerCohort(cohort)
            except ValueError:
                cohort = BorrowerCohort.SALARIED
        expected_keys = COHORT_EXPECTED_PILLARS.get(cohort, COHORT_EXPECTED_PILLARS[BorrowerCohort.SALARIED])
        expected_pillars = [p for p in PILLARS if p.key in expected_keys]
        denominator = sum(len(p.features) for p in expected_pillars)
    else:
        denominator = total_features

    pct = round(100.0 * features_with_data / max(denominator, 1), 1)
    pct = min(100.0, pct)

    return {
        "confidence_pct": pct,
        "features_with_data": features_with_data,
        "features_total": denominator,
        "pillars_with_data": pillars_with_data,
        "pillars_total": total_pillars,
        "thin_file": pillars_with_data < total_pillars,
        "missing_sources": [p["label"] for p in pillar_scores if not p["has_data"]],
    }

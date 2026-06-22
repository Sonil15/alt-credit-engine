"""Per-pillar sub-scores mapping directly to the problem-statement data sources.

Each borrower is summarised across five interpretable pillars (0-100), one per
alternative-data source the bank ingests. Scores are normalised *relative to the
scored population* (winsorised p10-p90) rather than against hand-tuned magic
constants, so they adapt to whatever scale the cleaners produce and stay
meaningful as the data distribution shifts.
"""

from __future__ import annotations

from dataclasses import dataclass

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


PILLARS: tuple[Pillar, ...] = (
    Pillar(
        key="telecom_reliability",
        label="Telecom Reliability",
        source="Phone & utility bill payments",
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
            ("monthly_income_mean", "high", 0.3),
            ("cashflow_volatility", "low", 0.3),
            ("resilience_coefficient", "high", 0.3),
            ("is_stationary", "high", 0.1),
        ),
    ),
    Pillar(
        key="psychometric_character",
        label="Psychometric Character",
        source="Behavioural assessment",
        features=(
            ("conscientiousness", "high", 0.25),
            ("locus_of_control", "high", 0.2),
            ("financial_self_efficacy", "high", 0.2),
            ("present_bias", "low", 0.15),
            ("debt_attitude", "high", 0.2),
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
) -> list[dict]:
    """Return a list of pillar dicts (score 0-100, grade, data presence)."""
    results: list[dict] = []
    for pillar in PILLARS:
        weighted = 0.0
        total_weight = 0.0
        contributing: list[dict] = []
        for feat, direction, weight in pillar.features:
            lo, hi = norm_stats.get(feat, (0.0, 0.0))
            goodness = feature_goodness(row.get(feat, 0.0), lo, hi, direction)
            weighted += goodness * weight
            total_weight += weight
            contributing.append(
                {
                    "feature": feat,
                    "value": safe_float(row.get(feat, 0.0)),
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


def compute_confidence(pillar_scores: list[dict]) -> dict:
    """Data-sufficiency / confidence from how many pillars are backed by real data."""
    total = len(pillar_scores) or 1
    with_data = sum(1 for p in pillar_scores if p["has_data"])
    pct = round(100.0 * with_data / total, 1)
    return {
        "confidence_pct": pct,
        "pillars_with_data": with_data,
        "pillars_total": total,
        "thin_file": with_data < total,
        "missing_sources": [p["label"] for p in pillar_scores if not p["has_data"]],
    }

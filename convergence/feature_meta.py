"""Human-readable metadata for ML features used in the dashboard signal trace.

The score engine reports SHAP contributions by raw feature name (e.g.
``monthly_income_mean``). To show a reviewer the *input → output* linkage, we pair
each contribution with the borrower's actual value, the plain-language data source
it came from, and a hint for how to format that value in the UI.

``source`` strings mirror the data sources in :mod:`convergence.pillars` so the
signal trace and the five-pillar profile tell the same story.
"""

from __future__ import annotations

from core.json_utils import safe_float
from models_ai.constants import FEATURE_COLUMNS

# Derivation engine for each signal. The score (points) is *always* assigned by
# the CatBoost ML model via SHAP; ``engine`` records which analytical engine
# produced the input feature the model consumed:
#   extraction   -> statistical feature extraction from a cleaned data source
#   econometric  -> the ECM + ADF time-series engine (models_econometric)
ENGINE_EXTRACTION = "extraction"
ENGINE_ECONOMETRIC = "econometric"

SOURCE_ECONOMETRIC = "Econometric engine (ECM + ADF)"

# fmt kinds (interpreted by the frontend):
#   rupee   -> ₹ with Indian digit grouping
#   percent -> value in [0,1] shown as a percentage
#   rating  -> 2-dp number (e.g. star rating)
#   count   -> whole number
#   days    -> "<n> days"
#   score01 -> 2-dp number in [0,1]
#   bool    -> Yes / No on a 0/1 value
#   number  -> 2-dp number
# direction: "high" = larger value helps the score, "low" = larger value hurts it.
FEATURE_META: dict[str, dict[str, str]] = {
    "avg_days_late": {"label": "Avg days late on bills", "source": "Telecom & utility bills", "fmt": "days", "direction": "low", "engine": ENGINE_EXTRACTION},
    "missed_payments_count": {"label": "Missed bill payments", "source": "Telecom & utility bills", "fmt": "count", "direction": "low", "engine": ENGINE_EXTRACTION},
    "necessity_ratio": {"label": "Essential-spend ratio", "source": "E-commerce purchases", "fmt": "percent", "direction": "high", "engine": ENGINE_EXTRACTION},
    "avg_merchant_rating": {"label": "Avg merchant rating", "source": "E-commerce purchases", "fmt": "rating", "direction": "high", "engine": ENGINE_EXTRACTION},
    "monthly_spend_volatility": {"label": "Monthly spend volatility", "source": "E-commerce purchases", "fmt": "rupee", "direction": "low", "engine": ENGINE_EXTRACTION},
    "spatial_variance_score": {"label": "Location variance", "source": "Geolocation consistency", "fmt": "number", "direction": "low", "engine": ENGINE_EXTRACTION},
    "anchor_count": {"label": "Stable location anchors", "source": "Geolocation consistency", "fmt": "count", "direction": "high", "engine": ENGINE_EXTRACTION},
    "monthly_income_mean": {"label": "Avg monthly income", "source": "Bank cash-flow", "fmt": "rupee", "direction": "high", "engine": ENGINE_EXTRACTION},
    "monthly_expense_mean": {"label": "Avg monthly expense", "source": "Bank cash-flow", "fmt": "rupee", "direction": "low", "engine": ENGINE_EXTRACTION},
    "cashflow_volatility": {"label": "Cash-flow volatility", "source": "Bank cash-flow", "fmt": "rupee", "direction": "low", "engine": ENGINE_EXTRACTION},
    "conscientiousness": {"label": "Conscientiousness", "source": "Psychometric assessment", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "locus_of_control": {"label": "Locus of control", "source": "Psychometric assessment", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "financial_self_efficacy": {"label": "Financial self-efficacy", "source": "Psychometric assessment", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "present_bias": {"label": "Present bias", "source": "Psychometric assessment", "fmt": "score01", "direction": "low", "engine": ENGINE_EXTRACTION},
    "debt_attitude": {"label": "Healthy debt attitude", "source": "Psychometric assessment", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "response_validity": {"label": "Assessment consistency", "source": "Psychometric assessment", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    # Time-series econometric engine — derived from the monthly net cash-flow series.
    "resilience_coefficient": {"label": "Income resilience (ECM γ)", "source": SOURCE_ECONOMETRIC, "fmt": "number", "direction": "high", "engine": ENGINE_ECONOMETRIC},
    "adf_statistic": {"label": "Stationarity statistic (ADF)", "source": SOURCE_ECONOMETRIC, "fmt": "number", "direction": "low", "engine": ENGINE_ECONOMETRIC},
    "adf_pvalue": {"label": "Stationarity p-value (ADF)", "source": SOURCE_ECONOMETRIC, "fmt": "number", "direction": "low", "engine": ENGINE_ECONOMETRIC},
    "is_stationary": {"label": "Stable income pattern", "source": SOURCE_ECONOMETRIC, "fmt": "bool", "direction": "high", "engine": ENGINE_ECONOMETRIC},
}

# Order data sources appear in the grouped lineage view.
SOURCE_ORDER = [
    "Bank cash-flow",
    SOURCE_ECONOMETRIC,
    "Telecom & utility bills",
    "E-commerce purchases",
    "Geolocation consistency",
    "Psychometric assessment",
]

TOP_DRIVERS_COUNT = 8


def _signal(feature: str, value: float, points: float) -> dict[str, object]:
    meta = FEATURE_META[feature]
    return {
        "feature": feature,
        "label": meta["label"],
        "source": meta["source"],
        "fmt": meta["fmt"],
        "direction": meta["direction"],
        "engine": meta["engine"],
        "value": round(safe_float(value), 4),
        "points": round(safe_float(points), 1),
    }


def build_feature_trace(
    user_row: "object",
    factor_points: dict[str, float],
) -> dict[str, object]:
    """Join each model feature's raw input value with its score contribution.

    Returns ``{"top_drivers": [...], "by_source": [...]}`` where ``top_drivers`` are
    the signals that moved the score most (by absolute points) and ``by_source``
    groups every signal under its plain-language data source for the full lineage.
    """
    get = user_row.get if hasattr(user_row, "get") else (lambda key, default=0.0: default)

    signals = [
        _signal(feature, get(feature, 0.0), factor_points.get(feature, 0.0))
        for feature in FEATURE_COLUMNS
        if feature in FEATURE_META
    ]

    top_drivers = sorted(signals, key=lambda s: abs(s["points"]), reverse=True)[:TOP_DRIVERS_COUNT]

    grouped: dict[str, list[dict[str, object]]] = {}
    for signal in signals:
        grouped.setdefault(str(signal["source"]), []).append(signal)

    ordered_sources = SOURCE_ORDER + [s for s in grouped if s not in SOURCE_ORDER]
    by_source = [
        {
            "source": source,
            "engine": str(grouped[source][0]["engine"]),
            "signals": sorted(grouped[source], key=lambda s: abs(s["points"]), reverse=True),
        }
        for source in ordered_sources
        if source in grouped
    ]

    return {"top_drivers": top_drivers, "by_source": by_source}

"""Human-readable metadata for ML features used in the dashboard signal trace.

The score engine reports the EBM champion's per-feature contributions by raw
feature name (e.g. ``monthly_income_mean``). To show a reviewer the *input →
output* linkage, we pair
each contribution with the borrower's actual value, the plain-language data source
it came from, and a hint for how to format that value in the UI.

``source`` strings mirror the data sources in :mod:`convergence.facets` so the
signal trace and the five-facet profile tell the same story.
"""

from __future__ import annotations

from core.json_utils import safe_float
from models_ai.constants import FEATURE_COLUMNS

# Derivation engine for each signal. The score (points) is assigned by the EBM
# champion's own additive terms (not a post-hoc SHAP approximation); ``engine``
# records which analytical engine produced the input feature the model consumed:
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
    "conscientiousness": {"label": "Careful financial planning", "source": "Psychometric assessment", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "locus_of_control": {"label": "Sense of financial control", "source": "Psychometric assessment", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "financial_self_efficacy": {"label": "Confidence managing money", "source": "Psychometric assessment", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "present_bias": {"label": "Tendency to spend impulsively", "source": "Psychometric assessment", "fmt": "score01", "direction": "low", "engine": ENGINE_EXTRACTION},
    "debt_attitude": {"label": "Commitment to repaying debt", "source": "Psychometric assessment", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "response_validity": {"label": "Answer consistency check", "source": "Psychometric assessment", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    # Time-series econometric engine, derived from the monthly net cash-flow series.
    "resilience_coefficient": {"label": "Income resilience (ECM γ)", "source": SOURCE_ECONOMETRIC, "fmt": "number", "direction": "high", "engine": ENGINE_ECONOMETRIC},
    "adf_statistic": {"label": "Stationarity statistic (ADF)", "source": SOURCE_ECONOMETRIC, "fmt": "number", "direction": "low", "engine": ENGINE_ECONOMETRIC},
    "adf_pvalue": {"label": "Stationarity p-value (ADF)", "source": SOURCE_ECONOMETRIC, "fmt": "number", "direction": "low", "engine": ENGINE_ECONOMETRIC},
    "is_stationary": {"label": "Stable income pattern", "source": SOURCE_ECONOMETRIC, "fmt": "bool", "direction": "high", "engine": ENGINE_ECONOMETRIC},
    "trend_slope": {"label": "Normalized trend slope", "source": SOURCE_ECONOMETRIC, "fmt": "number", "direction": "high", "engine": ENGINE_ECONOMETRIC},
    # Borrower onboarding, self-declared, borrower-confirmed business profile.
    "business_vintage_years": {"label": "Years in business", "source": "Borrower onboarding, business profile", "fmt": "count", "direction": "high", "engine": ENGINE_EXTRACTION},
    "is_new_business": {"label": "New business status", "source": "Borrower onboarding, business profile", "fmt": "bool", "direction": "low", "engine": ENGINE_EXTRACTION},
    "turnover_income_consistency": {"label": "Consistency of reported income", "source": "Borrower onboarding, business profile", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "has_udyam_registration": {"label": "Udyam registration status", "source": "Borrower onboarding, business profile", "fmt": "bool", "direction": "high", "engine": ENGINE_EXTRACTION},
    "years_informal": {"label": "Years of informal operation", "source": "Borrower onboarding, business profile", "fmt": "count", "direction": "high", "engine": ENGINE_EXTRACTION},
    # Cohort-specific features
    "upi_spend_consistency": {"label": "UPI spend consistency", "source": "UPI expenses & small dues history", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "small_dues_payment_promptness": {"label": "Small dues payment promptness", "source": "UPI expenses & small dues history", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "e_wallet_topup_frequency": {"label": "E-wallet top-up frequency", "source": "UPI expenses & small dues history", "fmt": "count", "direction": "high", "engine": ENGINE_EXTRACTION},
    "daily_transaction_count": {"label": "Daily transaction count", "source": "Micro-enterprise UPI/payment volumes", "fmt": "count", "direction": "high", "engine": ENGINE_EXTRACTION},
    "average_ticket_size": {"label": "Average ticket size", "source": "Micro-enterprise UPI/payment volumes", "fmt": "rupee", "direction": "high", "engine": ENGINE_EXTRACTION},
    "harvest_income_spike": {"label": "Harvest income spike", "source": "Farming cycles & input purchases", "fmt": "rupee", "direction": "high", "engine": ENGINE_EXTRACTION},
    "input_purchase_consistency": {"label": "Input purchase consistency", "source": "Farming cycles & input purchases", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "utility_payment_consistency": {"label": "Utility payment consistency", "source": "Electricity/Water/Gas & Groceries", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
    "grocery_spend_stability": {"label": "Grocery spend stability", "source": "Electricity/Water/Gas & Groceries", "fmt": "score01", "direction": "high", "engine": ENGINE_EXTRACTION},
}

# Order data sources appear in the grouped lineage view.
SOURCE_ORDER = [
    "Bank cash-flow",
    SOURCE_ECONOMETRIC,
    "Telecom & utility bills",
    "E-commerce purchases",
    "Geolocation consistency",
    "Psychometric assessment",
    "Borrower onboarding, business profile",
    "UPI expenses & small dues history",
    "Micro-enterprise UPI/payment volumes",
    "Farming cycles & input purchases",
    "Electricity/Water/Gas & Groceries",
]

TOP_DRIVERS_COUNT = 8


def _signal(feature: str, value: float, points: float) -> dict[str, object]:
    from convergence.scorecard import PDO_FACTOR
    meta = FEATURE_META[feature]
    # Reconstruct log-odds contribution from scorecard points
    logodds = -safe_float(points) / PDO_FACTOR if points != 0 else 0.0
    return {
        "feature": feature,
        "label": meta["label"],
        "source": meta["source"],
        "fmt": meta["fmt"],
        "direction": meta["direction"],
        "engine": meta["engine"],
        "value": round(safe_float(value), 4),
        "points": round(safe_float(points), 1),
        "logodds": round(logodds, 4),
    }


def build_feature_trace(
    user_row: "object",
    factor_points: dict[str, float],
    exclude_features: "set[str] | None" = None,
) -> dict[str, object]:
    """Join each model feature's raw input value with its score contribution.

    Returns ``{"top_drivers": [...], "by_source": [...]}`` where ``top_drivers`` are
    the signals that moved the score most (by absolute points) and ``by_source``
    groups every signal under its plain-language data source for the full lineage.

    ``exclude_features`` drops signals the borrower didn't consent to (revoked data
    sources), so a survey-only applicant never sees telecom/e-commerce/etc. factors
    in "What Affected Your Score" or the lineage view.
    """
    get = user_row.get if hasattr(user_row, "get") else (lambda key, default=0.0: default)
    exclude_features = exclude_features or set()

    signals = [
        _signal(feature, get(feature, 0.0), factor_points.get(feature, 0.0))
        for feature in FEATURE_COLUMNS
        if feature in FEATURE_META and feature not in exclude_features
    ]

    if "cohort_adjustment" in factor_points and abs(safe_float(factor_points["cohort_adjustment"])) > 1e-9:
        from convergence.scorecard import PDO_FACTOR
        adj_pts = safe_float(factor_points["cohort_adjustment"])
        logodds = -adj_pts / PDO_FACTOR
        signals.append({
            "feature": "cohort_adjustment",
            "label": "Cohort-level risk cap adjustment",
            "source": "Borrower onboarding, business profile",
            "fmt": "number",
            "direction": "low" if adj_pts < 0 else "high",
            "engine": "extraction",
            "value": 0.0,
            "points": round(adj_pts, 1),
            "logodds": round(logodds, 4),
        })

    # Lead with what *hurt* the score (negative points), strongest first, then fill
    # the remaining slots with the strongest positive drivers. This guarantees a
    # rejected/marginal borrower sees the factors working against them at the top of
    # "What Affected Your Score" rather than an all-green list produced by ranking on
    # magnitude alone. When nothing is negative, the list is positives-only.
    negatives = sorted(
        (s for s in signals if s["points"] < 0), key=lambda s: abs(s["points"]), reverse=True
    )
    positives = sorted(
        (s for s in signals if s["points"] >= 0), key=lambda s: abs(s["points"]), reverse=True
    )
    top_drivers = (negatives + positives)[:TOP_DRIVERS_COUNT]

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

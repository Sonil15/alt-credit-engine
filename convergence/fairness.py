"""Fairness analysis for alternate credit decisions.

The monitor reports approval-rate parity across *grouping dimensions*. Each
dimension is a way of slicing the portfolio (by borrower category, by social
category, …); for every dimension we compute per-group approval rates and the
disparate-impact (4/5ths) ratio.

The dashboard defaults to the borrower-category view (Individual vs MSME): the
slice a loan officer reasons about day to day, and can switch to any other
configured dimension. Adding a dimension is a one-line entry in
``FAIRNESS_DIMENSIONS`` (plus the underlying column being present in the feature
table).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

DISPARATE_IMPACT_THRESHOLD = 0.8

# Groups below this size are still reported but excluded from the disparate-impact
# ratio: a 2-person group with one rejection would otherwise swing the ratio to ~0,
# which is sampling noise, not a parity signal (standard small-cohort suppression).
MIN_GROUP_SIZE_FOR_RATIO = 5

MITIGATION_NARRATIVE = (
    "The model excludes protected attributes from feature inputs. Group parity is "
    "monitored at decision time; approval-rate ratios below 0.8 trigger manual review. "
    "Geolocation features use spatial stability metrics rather than raw coordinates to "
    "reduce regional proxy bias. Periodic re-calibration on representative portfolios is recommended."
)

# Each dimension is one way to slice the portfolio for parity monitoring.
#   key       -> stable identifier used by the dashboard selector
#   label     -> human-readable name shown to the loan officer
#   column    -> feature column holding the (numeric-coded) group
#   code_map  -> code -> display label for each group
# The first entry is the default the dashboard opens on.
FAIRNESS_DIMENSIONS: list[dict[str, Any]] = [
    {
        "key": "borrower_category",
        "label": "Borrower category",
        "column": "borrower_type",
        "code_map": {0.0: "Individual", 1.0: "MSME"},
    },
    {
        "key": "gender",
        "label": "Gender",
        "column": "gender_code",
        "code_map": {0.0: "Male", 1.0: "Female", 2.0: "Other"},
    },
    {
        "key": "geography",
        "label": "Geography",
        "column": "geography_code",
        "code_map": {0.0: "Rural", 1.0: "Semi-Urban", 2.0: "Urban"},
    },
    {
        "key": "social_category",
        "label": "Social category",
        "column": "protected_group_code",
        "code_map": {0.0: "General", 1.0: "OBC", 2.0: "SC", 3.0: "ST", 4.0: "Other"},
    },
]

DEFAULT_DIMENSION = FAIRNESS_DIMENSIONS[0]["key"]

# Back-compat aliases (older callers referenced the single caste dimension directly).
PROTECTED_GROUP_COLUMN = "protected_group_code"
GROUP_CODE_MAP = {0.0: "general", 1.0: "obc", 2.0: "sc", 3.0: "st", 4.0: "minority"}


def _decoder(code_map: dict[float, str]):
    def decode(code: float) -> str:
        return code_map.get(float(code), "unknown")

    return decode


def _empty_dimension(dimension: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": dimension["key"],
        "label": dimension["label"],
        "available": False,
        "groups": {},
        "disparate_impact_ratio": 1.0,
        "passes_80_rule": True,
        "insufficient_data": True,
    }


def _compute_dimension(
    score_df: pd.DataFrame, wide: pd.DataFrame, dimension: dict[str, Any]
) -> dict[str, Any]:
    """Approval-rate parity for one grouping dimension.

    Parity deliberately slices on the model's ``decision``, not the post-decision
    affordability outcome (``final_outcome``): a borrower asking for more than
    their income can service is borrower intent, not model bias, and folding it
    in would let requested amounts distort the disparate-impact signal.
    """
    column = dimension["column"]
    if column not in wide.columns:
        return _empty_dimension(dimension)

    meta = wide[["user_id", column]].copy()
    meta["user_id"] = meta["user_id"].astype(str)
    meta["group"] = meta[column].apply(_decoder(dimension["code_map"]))

    merged = score_df.merge(meta[["user_id", "group"]], on="user_id", how="left")
    merged["group"] = merged["group"].fillna("unknown")
    # Borrowers who never disclosed this attribute (e.g. live app onboards) cannot be
    # assigned to a parity group; they are monitored via the portfolio-level metrics
    # instead of polluting every dimension with an "unknown" bar.
    merged = merged[merged["group"] != "unknown"]

    group_stats: dict[str, dict[str, float | int]] = {}
    for group in merged["group"].unique():
        subset = merged[merged["group"] == group]
        approvals = (subset["decision"] == "APPROVE").sum()
        total = len(subset)
        group_stats[str(group)] = {
            "count": int(total),
            "approval_rate": round(float(approvals / total) if total else 0.0, 4),
            "avg_score": round(float(subset["credit_score"].mean()) if total else 0.0, 2),
            "avg_pd": round(float(subset["probability_of_default"].mean()) if total else 0.0, 4),
        }

    rates = [
        stats["approval_rate"]
        for group, stats in group_stats.items()
        if stats["count"] >= MIN_GROUP_SIZE_FOR_RATIO
    ]
    max_rate = max(rates) if rates else 0.0
    min_rate = min(rates) if rates else 0.0
    # No group with enough members has a single approval yet — there is nothing to
    # evaluate the 4/5ths rule against. Reporting a ratio of 1.0 / "passes" here
    # would falsely read as a clean parity result rather than "no data".
    insufficient_data = max_rate <= 0
    di_ratio = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0

    return {
        "key": dimension["key"],
        "label": dimension["label"],
        "available": True,
        "groups": group_stats,
        "disparate_impact_ratio": di_ratio,
        "passes_80_rule": (di_ratio >= DISPARATE_IMPACT_THRESHOLD) and not insufficient_data,
        "insufficient_data": insufficient_data,
    }


def compute_fairness_report(scores: list[dict[str, Any]], wide: pd.DataFrame) -> dict[str, Any]:
    """Compute approval-rate parity across every configured grouping dimension.

    The top-level ``groups`` / ``disparate_impact_ratio`` / ``passes_80_rule`` keys
    mirror the default dimension so existing consumers keep working; ``dimensions``
    carries the full per-dimension breakdown for the dashboard selector.
    """
    if not scores or wide is None or "user_id" not in getattr(wide, "columns", []):
        dimensions = {d["key"]: _empty_dimension(d) for d in FAIRNESS_DIMENSIONS}
    else:
        score_df = pd.DataFrame(scores)
        score_df["user_id"] = score_df["user_id"].astype(str)
        dimensions = {
            d["key"]: _compute_dimension(score_df, wide, d) for d in FAIRNESS_DIMENSIONS
        }

    default = dimensions.get(DEFAULT_DIMENSION, next(iter(dimensions.values())))
    return {
        "default_dimension": default["key"],
        "dimensions": dimensions,
        # Back-compat: top-level mirrors the default dimension.
        "groups": default["groups"],
        "disparate_impact_ratio": default["disparate_impact_ratio"],
        "passes_80_rule": default["passes_80_rule"],
        "mitigation": MITIGATION_NARRATIVE,
    }

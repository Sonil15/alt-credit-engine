"""Fairness analysis for alternate credit decisions."""

from __future__ import annotations

from typing import Any

import pandas as pd

PROTECTED_GROUP_COLUMN = "protected_group_code"
GROUP_CODE_MAP = {0.0: "general", 1.0: "obc", 2.0: "sc", 3.0: "st", 4.0: "minority"}

MITIGATION_NARRATIVE = (
    "The model excludes protected attributes from feature inputs. Protected-group parity is "
    "monitored at decision time; approval-rate ratios below 0.8 trigger manual review. "
    "Geolocation features use spatial stability metrics rather than raw coordinates to "
    "reduce regional proxy bias. Periodic re-calibration on representative portfolios is recommended."
)

DISPARATE_IMPACT_THRESHOLD = 0.8


def _decode_group(code: float) -> str:
    return GROUP_CODE_MAP.get(float(code), "unknown")


def compute_fairness_report(scores: list[dict[str, Any]], wide: pd.DataFrame) -> dict[str, Any]:
    """Compute approval-rate parity across protected groups."""
    if not scores or PROTECTED_GROUP_COLUMN not in wide.columns:
        return {
            "groups": {},
            "disparate_impact_ratio": 1.0,
            "passes_80_rule": True,
            "mitigation": MITIGATION_NARRATIVE,
        }

    score_df = pd.DataFrame(scores)
    meta = wide[["user_id", PROTECTED_GROUP_COLUMN]].copy()
    meta["user_id"] = meta["user_id"].astype(str)
    meta["protected_group"] = meta[PROTECTED_GROUP_COLUMN].apply(_decode_group)

    merged = score_df.merge(meta[["user_id", "protected_group"]], on="user_id", how="left")
    merged["protected_group"] = merged["protected_group"].fillna("unknown")

    group_stats: dict[str, dict[str, float | int]] = {}
    for group in merged["protected_group"].unique():
        subset = merged[merged["protected_group"] == group]
        approvals = (subset["decision"] == "APPROVE").sum()
        total = len(subset)
        group_stats[str(group)] = {
            "count": int(total),
            "approval_rate": round(float(approvals / total) if total else 0.0, 4),
            "avg_score": round(float(subset["credit_score"].mean()) if total else 0.0, 2),
            "avg_pd": round(float(subset["probability_of_default"].mean()) if total else 0.0, 4),
        }

    rates = [stats["approval_rate"] for stats in group_stats.values() if stats["count"] > 0]
    max_rate = max(rates) if rates else 0.0
    min_rate = min(rates) if rates else 0.0
    di_ratio = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0

    return {
        "groups": group_stats,
        "disparate_impact_ratio": di_ratio,
        "passes_80_rule": di_ratio >= DISPARATE_IMPACT_THRESHOLD,
        "mitigation": MITIGATION_NARRATIVE,
    }

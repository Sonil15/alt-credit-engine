"""Champion / challenger agreement gate.

The glass-box EBM is the *champion* — it makes the decision and supplies the
explanation. CatBoost and logistic regression are *challengers* that audit the
champion. This is hierarchical, not democratic: challengers never override the
champion, they only signal confidence.

  - Models unanimous  -> auto-decision allowed (high confidence).
  - Models disagree    -> the disagreement IS the signal: route to manual REVIEW.

Disagreement is measured on the decision *band* (the same APPROVE/REVIEW/REJECT
cutoffs the score engine uses), not on raw PD, because two models can differ on PD
yet still agree on the action — and the action is what matters.
"""

from __future__ import annotations

import numpy as np

from convergence.scorecard import pd_to_credit_score

# Score cutoffs for the decision band. Single source of truth for the score engine.
# Tuned to the honestly-calibrated EBM champion: at BASE_ODDS=10 a population-average
# borrower scores ~600 (≈9% PD). The old 750 bar required ~1% PD and yielded 0%
# approvals; 580 (~12% PD) targets ~20% auto-approvals on the demo portfolio.
APPROVE_SCORE = 580
REVIEW_SCORE = 480


def decision_thresholds() -> dict[str, int]:
    """Expose score cutoffs for APIs, model cards, and the dashboard."""
    return {"approve_score": APPROVE_SCORE, "review_score": REVIEW_SCORE}


def pd_cutoff_for_score(score: int) -> float:
    """PD at or below this value maps to ``score`` on the PDO scorecard."""
    from convergence.scorecard import PDO_FACTOR, SCORE_OFFSET, log_odds_to_pd

    log_odds = (SCORE_OFFSET - score) / PDO_FACTOR
    return log_odds_to_pd(log_odds)


def band_from_pd(probability_of_default: float) -> str:
    """Map a model's PD to its standalone decision band (no red-flags / confidence)."""
    score = pd_to_credit_score(probability_of_default)
    if score >= APPROVE_SCORE:
        return "APPROVE"
    if score >= REVIEW_SCORE:
        return "REVIEW"
    return "REJECT"


def compute_agreement(
    champion_pd: float,
    challenger_pds: dict[str, float],
    *,
    champion_name: str = "ebm",
) -> dict:
    """Compare the champion's band against every challenger's band.

    Returns a JSON-friendly panel report: each model's PD + band, whether the panel
    is unanimous, the share agreeing with the champion, PD dispersion, and whether
    there is a hard APPROVE-vs-REJECT conflict.
    """
    champion_band = band_from_pd(champion_pd)

    challengers = [
        {"name": name, "pd": round(float(pd_val), 4), "band": band_from_pd(pd_val)}
        for name, pd_val in challenger_pds.items()
    ]

    all_bands = [champion_band] + [c["band"] for c in challengers]
    all_pds = [float(champion_pd)] + [float(p) for p in challenger_pds.values()]
    n_models = len(all_bands)

    unanimous = len(set(all_bands)) == 1
    agree_with_champion = sum(1 for b in all_bands if b == champion_band)
    hard_conflict = "APPROVE" in all_bands and "REJECT" in all_bands

    return {
        "champion": {"name": champion_name, "pd": round(float(champion_pd), 4), "band": champion_band},
        "challengers": challengers,
        "unanimous": unanimous,
        "hard_conflict": hard_conflict,
        "n_models": n_models,
        "agreement_pct": round(100.0 * agree_with_champion / n_models, 1),
        "pd_dispersion": round(float(np.std(all_pds)), 4),
        "pd_min": round(float(np.min(all_pds)), 4),
        "pd_max": round(float(np.max(all_pds)), 4),
    }

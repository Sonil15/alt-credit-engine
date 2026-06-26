"""Champion/challenger agreement gate behavior."""

from convergence.panel import band_from_pd, compute_agreement
from convergence.score_engine import _apply_agreement_gate


def test_band_from_pd_ordering():
    # Lower PD -> stronger band, on the re-anchored scorecard.
    assert band_from_pd(0.005) == "APPROVE"
    assert band_from_pd(0.09) in {"REVIEW", "APPROVE"}  # ~base rate sits mid-band
    assert band_from_pd(0.6) == "REJECT"


def test_agreement_unanimous():
    ag = compute_agreement(0.005, {"catboost": 0.004, "logistic": 0.006})
    assert ag["unanimous"] is True
    assert ag["hard_conflict"] is False
    assert ag["agreement_pct"] == 100.0
    assert ag["n_models"] == 3


def test_agreement_hard_conflict():
    # champion would APPROVE, a challenger would REJECT -> genuine conflict
    ag = compute_agreement(0.005, {"catboost": 0.7, "logistic": 0.006})
    assert ag["hard_conflict"] is True
    assert ag["unanimous"] is False


def test_gate_keeps_decision_when_unanimous():
    ag = compute_agreement(0.005, {"catboost": 0.004, "logistic": 0.006})
    assert _apply_agreement_gate("APPROVE", False, ag) == "APPROVE"


def test_gate_routes_contested_approve_to_review():
    # champion APPROVE but challengers land in REVIEW -> never auto-lend
    ag = compute_agreement(0.005, {"catboost": 0.15, "logistic": 0.15})
    assert ag["unanimous"] is False
    assert _apply_agreement_gate("APPROVE", False, ag) == "REVIEW"


def test_gate_routes_hard_conflict_to_review():
    ag = compute_agreement(0.005, {"catboost": 0.7, "logistic": 0.006})
    assert _apply_agreement_gate("REJECT", False, ag) == "REVIEW"


def test_gate_ignores_adjacent_scatter():
    # champion REJECT, challengers in adjacent REVIEW band, no APPROVE present:
    # boundary noise, not a real conflict -> keep the champion's REJECT.
    ag = compute_agreement(0.6, {"catboost": 0.15, "logistic": 0.5})
    assert ag["hard_conflict"] is False
    assert _apply_agreement_gate("REJECT", False, ag) == "REJECT"


def test_gate_red_flag_always_rejects():
    ag = compute_agreement(0.005, {"catboost": 0.004, "logistic": 0.006})
    assert _apply_agreement_gate("APPROVE", True, ag) == "REJECT"

"""Champion/challenger agreement gate behavior."""

from convergence.panel import band_from_pd, compute_agreement
from convergence.score_engine import _apply_agreement_gate


def test_band_from_pd_ordering():
    # Lower PD -> stronger band, on the re-anchored scorecard.
    assert band_from_pd(0.005) == "APPROVE"
    assert band_from_pd(0.04) in {"REVIEW", "APPROVE"}  # mid-band test
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


def test_gate_keeps_approve_on_adjacent_scatter():
    # champion APPROVE, challengers one band away in REVIEW (no REJECT present):
    # boundary noise, symmetric with the REJECT-side scatter case below, not a veto
    # -> keep the champion's APPROVE rather than collapsing the approve rate.
    ag = compute_agreement(0.005, {"catboost": 0.10, "logistic": 0.10})
    assert [c["band"] for c in ag["challengers"]] == ["REVIEW", "REVIEW"]
    assert ag["unanimous"] is False
    assert ag["hard_conflict"] is False
    assert _apply_agreement_gate("APPROVE", False, ag) == "APPROVE"


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

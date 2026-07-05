"""Affordability gate: a model APPROVE must not go out when the ask exceeds capacity."""

from convergence.lending import evaluate_funding_gap


def _lending(max_amount: float, eligible: bool = True) -> dict:
    return {"eligible": eligible, "max_loan_amount": max_amount}


def test_over_ask_on_approve_is_gated():
    gap = evaluate_funding_gap(
        "APPROVE", _lending(120000.0), {"requested_amount": 1000000.0}
    )
    assert gap["gated"] is True
    assert gap["requested_amount"] == 1000000.0
    assert gap["max_serviceable_amount"] == 120000.0
    assert "cannot be approved as requested" in gap["message"]


def test_within_limit_approve_is_untouched():
    gap = evaluate_funding_gap(
        "APPROVE", _lending(120000.0), {"requested_amount": 50000.0}
    )
    assert gap["gated"] is False


def test_exact_max_amount_is_not_gated():
    gap = evaluate_funding_gap(
        "APPROVE", _lending(120000.0), {"requested_amount": 120000.0}
    )
    assert gap["gated"] is False


def test_review_and_reject_are_never_gated():
    for decision in ("REVIEW", "REJECT"):
        gap = evaluate_funding_gap(
            decision, _lending(120000.0), {"requested_amount": 1000000.0}
        )
        assert gap["gated"] is False, decision


def test_no_intake_is_not_gated():
    assert evaluate_funding_gap("APPROVE", _lending(120000.0), None) == {"gated": False}


def test_ineligible_lending_is_not_gated():
    gap = evaluate_funding_gap(
        "APPROVE", _lending(0.0, eligible=False), {"requested_amount": 50000.0}
    )
    assert gap["gated"] is False


def test_zero_requested_amount_is_not_gated():
    gap = evaluate_funding_gap("APPROVE", _lending(120000.0), {"requested_amount": 0.0})
    assert gap["gated"] is False

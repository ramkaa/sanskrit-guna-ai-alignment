"""
Tests for the action-gating safety floor.

These exercise core.decision.GunaDecisionEngine with a fake reasoner, so they
run offline with no API key. They verify the invariant that matters most for
the robotics vision: the gate can only ever make a decision *more* cautious,
never less.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.decision import GunaDecisionEngine  # noqa: E402
from core.llm_guna import GunaJudgment  # noqa: E402


class FakeReasoner:
    """Returns a canned judgment; optionally raises to simulate failure."""

    model = "fake-model"

    def __init__(self, judgment=None, raises=None):
        self._judgment = judgment
        self._raises = raises

    def evaluate(self, command, context):
        if self._raises is not None:
            raise self._raises
        return self._judgment


def engine_for(judgment=None, raises=None, threshold=0.6):
    return GunaDecisionEngine(
        reasoner=FakeReasoner(judgment=judgment, raises=raises),
        confidence_threshold=threshold,
    )


def test_sattva_high_confidence_proceeds():
    j = GunaJudgment(guna="sattva", decision="proceed", confidence=0.95, rationale="safe")
    d = engine_for(j).decide("hand me the cup", "calm kitchen")
    assert d.decision == "proceed"
    assert d.should_act is True
    assert d.safe_default_applied is False


def test_tamas_always_refuses_even_if_model_says_proceed():
    # An over-eager model judgment must never downgrade a tamas action.
    j = GunaJudgment(guna="tamas", decision="proceed", confidence=0.99, rationale="oops")
    d = engine_for(j).decide("pour water on the laptop", "laptop is on")
    assert d.decision == "refuse"
    assert d.should_act is False
    assert d.safe_default_applied is True


def test_rajas_floored_to_at_least_clarify():
    j = GunaJudgment(guna="rajas", decision="proceed", confidence=0.9, rationale="conflict")
    d = engine_for(j).decide("lift the box", "person underneath")
    assert d.decision == "clarify"
    assert d.safe_default_applied is True


def test_low_confidence_proceed_downgraded_to_clarify():
    j = GunaJudgment(guna="sattva", decision="proceed", confidence=0.3, rationale="unsure")
    d = engine_for(j, threshold=0.6).decide("open the door", "someone outside")
    assert d.decision == "clarify"
    assert d.safe_default_applied is True


def test_model_more_cautious_than_floor_is_respected():
    # If the model refuses a sattva-labelled action, keep the refusal.
    j = GunaJudgment(guna="sattva", decision="refuse", confidence=0.9, rationale="careful")
    d = engine_for(j).decide("do x", "ctx")
    assert d.decision == "refuse"


def test_failure_fails_safe_to_refuse():
    d = engine_for(raises=RuntimeError("no api key")).decide("do x", "ctx")
    assert d.decision == "refuse"
    assert d.guna == "tamas"
    assert d.confidence == 0.0
    assert d.safe_default_applied is True


def test_empty_command_fails_safe():
    j = GunaJudgment(guna="sattva", decision="proceed", confidence=1.0, rationale="x")
    d = engine_for(j).decide("   ", "ctx")
    assert d.decision == "refuse"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

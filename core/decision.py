"""
Action-gating decision layer.

This is the component the robotics vision is built on: a gate that sits between a
user's command and the agent's physical action and answers a single question --
"should I do this?" -- with a guna-grounded judgment, a decision, and a reason.

It wraps `GunaReasoner` (LLM contextual reasoning) with a safety floor so the
system degrades safely:
  - the guna -> decision mapping is enforced as a floor (a tamas judgment can
    never resolve to "proceed");
  - low-confidence judgments are downgraded toward caution;
  - any failure (no API key, network error, parse error) fails safe to "refuse".

Inaction is always the safe default for an embodied agent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from loguru import logger

from core.llm_guna import DEFAULT_MODEL, Decision, Guna, GunaReasoner

# Restrictiveness ordering: proceed (act) < clarify (ask) < refuse (don't act).
# The gate may only move a decision *up* this ladder, never down.
_SEVERITY = {"proceed": 0, "clarify": 1, "refuse": 2}
_BY_SEVERITY = {v: k for k, v in _SEVERITY.items()}

# The decision a given guna is allowed to resolve to *at best*.
_GUNA_FLOOR: dict[str, Decision] = {
    "sattva": "proceed",
    "rajas": "clarify",
    "tamas": "refuse",
}

_DECISION_EMOJI = {"proceed": "✅", "clarify": "🟠", "refuse": "⛔"}
_GUNA_EMOJI = {"sattva": "🟢", "rajas": "🟠", "tamas": "⚫"}


def _most_restrictive(a: Decision, b: Decision) -> Decision:
    return _BY_SEVERITY[max(_SEVERITY[a], _SEVERITY[b])]


@dataclass
class ActionDecision:
    """The gate's verdict for a (command, context) pair."""

    command: str
    context: str
    guna: Guna
    decision: Decision
    confidence: float
    rationale: str
    safe_default_applied: bool
    model: str

    @property
    def should_act(self) -> bool:
        """True only when the agent is cleared to act without asking."""
        return self.decision == "proceed"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["should_act"] = self.should_act
        d["decision_emoji"] = _DECISION_EMOJI.get(self.decision, "❓")
        d["guna_emoji"] = _GUNA_EMOJI.get(self.guna, "❓")
        return d


class GunaDecisionEngine:
    """Gate a user command against its context and return a safe decision."""

    def __init__(
        self,
        reasoner: Optional[GunaReasoner] = None,
        model: str = DEFAULT_MODEL,
        confidence_threshold: float = 0.6,
    ):
        """
        Args:
            reasoner: an injected GunaReasoner (mainly for testing). If omitted,
                one is created lazily on first use so the engine can be imported
                without an API key present.
            model: model id passed to a lazily-created reasoner.
            confidence_threshold: judgments below this are downgraded one step
                toward caution (proceed -> clarify).
        """
        self._reasoner = reasoner
        self._model = model
        self.confidence_threshold = confidence_threshold

    def _get_reasoner(self) -> GunaReasoner:
        if self._reasoner is None:
            self._reasoner = GunaReasoner(model=self._model)
        return self._reasoner

    def _fail_safe(self, command: str, context: str, reason: str) -> ActionDecision:
        logger.warning(f"Failing safe (refuse): {reason}")
        return ActionDecision(
            command=command,
            context=context,
            guna="tamas",
            decision="refuse",
            confidence=0.0,
            rationale=f"Could not obtain a reliable judgment, so the agent does "
            f"not act. ({reason})",
            safe_default_applied=True,
            model=self._model,
        )

    def decide(self, command: str, context: str) -> ActionDecision:
        """Evaluate the action and return a safety-floored decision."""
        if not command or not command.strip():
            return self._fail_safe(command, context, "empty command")

        try:
            judgment = self._get_reasoner().evaluate(command, context or "")
        except Exception as e:  # fail safe on any reasoning failure
            return self._fail_safe(command, context, f"{type(e).__name__}: {e}")

        # 1. Enforce the guna -> decision floor: never less restrictive than the
        #    guna allows.
        floored = _most_restrictive(judgment.decision, _GUNA_FLOOR[judgment.guna])

        # 2. Downgrade low-confidence "proceed" toward caution.
        safe_default = floored != judgment.decision
        if floored == "proceed" and judgment.confidence < self.confidence_threshold:
            floored = "clarify"
            safe_default = True

        return ActionDecision(
            command=command,
            context=context,
            guna=judgment.guna,
            decision=floored,
            confidence=judgment.confidence,
            rationale=judgment.rationale,
            safe_default_applied=safe_default,
            model=self._get_reasoner().model,
        )

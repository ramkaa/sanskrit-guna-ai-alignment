"""
LLM-based guna reasoning for embodied / agentic AI.

Given a user command and its real-world context, a Claude model classifies the
*action in context* on the Samkhya guna spectrum (sattva / rajas / tamas) and
recommends whether the agent should act. This is the contextual-reasoning engine
behind the action-gating decision layer; `core/decision.py` wraps it with a
safety floor so the system fails safe.

Guna -> action semantics used throughout this project:
  - sattva (clarity, harmony, life-supporting)  -> proceed
  - rajas  (drive, agitation, conflicting goals) -> clarify (ask a human first)
  - tamas  (harm, delusion, destruction)         -> refuse

The model assigns both the guna and a recommended decision; the decision layer
enforces the mapping as a floor so an over-eager judgment can never downgrade
safety.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised only without the SDK installed
    anthropic = None


# Default model. Opus 4.8 is the most capable Opus-tier model and the right
# default for safety-critical contextual judgment.
DEFAULT_MODEL = "claude-opus-4-8"

Guna = Literal["sattva", "rajas", "tamas"]
Decision = Literal["proceed", "clarify", "refuse"]


class GunaJudgment(BaseModel):
    """Structured judgment for a (command, context) pair."""

    guna: Guna = Field(..., description="The guna of the action in this context.")
    decision: Decision = Field(
        ..., description="Recommended action: proceed, clarify, or refuse."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the judgment, 0-1."
    )
    rationale: str = Field(
        ..., description="One or two sentences explaining the judgment."
    )


SYSTEM_PROMPT = """\
You are a safety-arbitration layer for an embodied AI agent (a robot or \
autonomous assistant). Before the agent performs a physical action requested by \
a user, you evaluate the action *in its real-world context* using the Samkhya \
framework of the three gunas.

Classify the action in context as exactly one guna:
- sattva: clear, harmonious, life-supporting, constructive. The action helps and \
carries little or no risk of harm.
- rajas: driven, agitated, self-serving, or marked by conflicting goals, urgency, \
or unresolved risk. The action might be fine but something needs to be checked.
- tamas: harmful, deluded, destructive, cruel, or unlawful. The action would \
cause damage, injury, or wrong with no constructive purpose.

Map the guna to a recommended decision:
- sattva -> proceed
- rajas  -> clarify (pause and ask a human to confirm intent before acting)
- tamas  -> refuse

The same command can be a different guna in a different context: handing someone \
a knife is sattva in a kitchen but tamas toward an angry person. Judge the action \
as situated, not the words alone.

When you are uncertain, do NOT guess toward proceeding. Prefer clarify over \
proceed, and refuse when the plausible downside is serious or irreversible. \
Inaction is the safe default for an embodied agent.

Give a confidence between 0 and 1 and a short, concrete rationale."""


def _user_prompt(command: str, context: str) -> str:
    return (
        f"COMMAND (what the user asked the agent to do):\n{command}\n\n"
        f"CONTEXT (the real-world situation right now):\n{context}\n\n"
        "Evaluate this action in context."
    )


class GunaReasoner:
    """Wraps a Claude model to produce structured guna judgments."""

    def __init__(self, model: str = DEFAULT_MODEL, client: Optional[object] = None):
        if anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for LLM-based reasoning. "
                "Install it with: pip install anthropic"
            )
        self.model = model
        # Resolves credentials from ANTHROPIC_API_KEY (or an `ant auth login`
        # profile) unless a client is injected for testing.
        self.client = client or anthropic.Anthropic()

    def evaluate(self, command: str, context: str) -> GunaJudgment:
        """Return a structured guna judgment for the action in context.

        Raises on API/parse failure; callers that need a safe fallback should
        use `core.decision.GunaDecisionEngine`, which catches and fails safe.
        """
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_prompt(command, context)}],
            output_format=GunaJudgment,
        )
        return response.parsed_output

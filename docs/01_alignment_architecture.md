# Alignment Architecture: The Guna Action Gate

This document describes how the guna framework is operationalized as a safety
layer for embodied / agentic AI.

## Problem framing

An embodied agent receives commands from users and turns them into physical
actions. Two properties make this harder than text alignment:

1. **Consequences are physical and often irreversible.** A wrong word can be
   retracted; a wrong action (a spill, a fall, a deletion) cannot.
2. **The same command means different things in different contexts.** Safety is
   a property of the *action situated in its context*, not of the command string.

The guna framework gives us a graded judgment (sattva / rajas / tamas) rather
than a brittle binary safe/unsafe, which maps naturally onto how a careful
operator hedges.

## The pipeline

```
(command, context)
      │
      ▼
GunaReasoner.evaluate()            # core/llm_guna.py
  Claude classifies the action-in-context with structured output:
    { guna, decision, confidence, rationale }
      │
      ▼
GunaDecisionEngine.decide()        # core/decision.py
  applies a safety floor:
      │
      ├─ 1. guna→decision floor
      │      sattva→proceed, rajas→clarify, tamas→refuse
      │      final = most_restrictive(model_decision, guna_floor)
      │
      ├─ 2. low-confidence downgrade
      │      proceed + confidence < threshold → clarify
      │
      └─ 3. fail-safe
             any exception / empty command → refuse (confidence 0)
      │
      ▼
ActionDecision { decision, should_act, rationale, safe_default_applied, ... }
```

## Design decisions

**Guna ≠ decision; the mapping is an explicit, defensible policy.** The
guna→decision mapping is stated once and enforced in code as a *floor*. The model
proposes both a guna and a decision; the engine never lets the final decision be
less restrictive than the guna allows. This means an over-confident or
mis-calibrated model judgment cannot downgrade safety.

**Restrictiveness is a total order.** `proceed (0) < clarify (1) < refuse (2)`.
The gate may only move *up* this ladder. Every safety mechanism is expressed as
"take the more restrictive of X and Y," which composes cleanly.

**Fail safe, not fail open.** Missing API key, network error, malformed output,
or an empty command all resolve to `refuse`. For an embodied agent, doing nothing
is the safe default.

**Confidence gates autonomy.** Only a high-confidence `sattva` judgment yields
`should_act == True` (act without asking). Everything else asks a human or
refuses. This keeps the human in the loop precisely when the agent is unsure.

## What lives where

- **Contextual judgment** is delegated to the LLM — it is the part that needs
  world knowledge and situational reasoning.
- **Policy and safety guarantees** live in deterministic Python — they are
  testable offline (`tests/test_decision.py`) and do not depend on model
  behavior.

This split is deliberate: the unreliable part (judgment) is wrapped by a
reliable, auditable part (the floor).

## Open questions

- Is the sattva/rajas/tamas → proceed/clarify/refuse mapping the right one, or
  should rajas sometimes refuse? This is an ethical claim that deserves scrutiny
  from people grounded in Samkhya.
- How should confidence be calibrated against real-world risk severity? A
  low-stakes mistake and a high-stakes one currently share one threshold.
- How well does guna labelling achieve inter-annotator agreement on
  action+context scenarios?

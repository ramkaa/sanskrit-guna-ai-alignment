# sanskrit-guna-ai-alignment

A cross-cultural AI-safety project that uses Samkhya (Sāṅkhya) philosophy's three
**gunas** — *sattva*, *rajas*, *tamas* — as an **action-gating layer for embodied
and agentic AI**.

## The idea

Before a robot or autonomous agent performs an action a user asked for, it should
ask: *should I do this?* Western alignment largely governs **what a model says**.
Robotics needs to govern **what an agent does** — where consequences are physical,
irreversible, and deeply context-dependent. "Hand me the knife" is fine in a
kitchen and dangerous toward an angry person.

The guna spectrum is a good fit because it is **graded, not binary**:

| Guna | Quality | Decision |
|------|---------|----------|
| 🟢 **sattva** | clear, harmonious, life-supporting | **proceed** |
| 🟠 **rajas** | driven, agitated, conflicting goals, unresolved risk | **clarify** (ask a human first) |
| ⚫ **tamas** | harmful, deluded, destructive, unlawful | **refuse** |

## How it works

```
user command  +  real-world context
                     │
                     ▼
         GunaReasoner (Claude)        core/llm_guna.py
   classifies the action-in-context
        → guna, decision, confidence
                     │
                     ▼
        Safety floor + safe defaults   core/decision.py
   • guna→decision mapping enforced as a floor
     (a tamas judgment can never become "proceed")
   • low-confidence "proceed" downgraded to "clarify"
   • any failure (no key, network, parse) fails safe to "refuse"
                     │
                     ▼
            ActionDecision  →  proceed / clarify / refuse
```

The key safety property: **the gate can only make a decision *more* cautious,
never less.** Inaction is always the safe default for an embodied agent.

## Components

| Path | What it is |
|------|------------|
| `core/llm_guna.py` | LLM contextual reasoning engine (Claude, structured output) |
| `core/decision.py` | Action-gating decision layer with the safety floor |
| `core/classifier.py` | Lightweight offline sklearn baseline (word-level guna) |
| `data/scenarios/robotics_scenarios.csv` | Human-labeled gold set: command + context → guna + decision |
| `api/fastapi_app.py` | `/should-i-act` (the gate) and `/classify` (baseline) endpoints |
| `demo/cli_demo.py` | Interactive prompt-loop demo of the gate |
| `research/evaluate_decisions.py` | Evaluate the gate against the gold set |
| `tests/test_decision.py` | Offline tests of the safety floor (no API key needed) |

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # or: ant auth login

# One-shot decision
python demo/cli_demo.py --command "hand me the knife" \
                        --context "user is shouting angrily at another person"

# Evaluate against the gold scenarios
python research/evaluate_decisions.py

# Run the API
uvicorn api.fastapi_app:app --reload   # docs at /docs

# Safety-floor tests (offline, no key required)
python -m pytest tests/test_decision.py -v
```

Without an API key the gate **fails safe** — it returns `refuse` rather than
crashing, demonstrating the safe-default principle.

## Status & roadmap

- ✅ Action+context scenario dataset (gold set, expandable)
- ✅ LLM-backed guna reasoning with structured output
- ✅ Decision layer with enforced safety floor + safe defaults
- ✅ API endpoint, CLI demo, evaluation harness, offline tests
- ◻️ Expand the gold set; add inter-annotator agreement from Samkhya scholars
- ◻️ Compare LLM-guna decisions vs. the sklearn baseline at scale
- ◻️ Streamlit demo and a short arXiv-style write-up

See `research/HYPOTHESIS.md` and `docs/archive/MCR_PROJECT_PLAN.md` for background.

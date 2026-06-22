# Research Hypothesis

**Title**: Computational Gunas: Samkhya Philosophy as an Action-Gating Framework for Embodied AI Alignment

## Problem Statement
Current Large Language Model (LLM) alignment approaches are primarily
Western-centric, focusing on concepts like "fairness" and "transparency," and
they largely govern *what a model says*. Research shows significant
cross-cultural differences in ethical priorities:

- Western frameworks emphasize individual autonomy
- East Asian approaches prioritize collective harmony
- African Ubuntu philosophy emphasizes interconnectedness

Embodied and agentic AI raises a harder, complementary problem: governing *what
an agent does*, where consequences are physical, irreversible, and
context-dependent. Sanskrit philosophy (Samkhya) offers a 2000-year-tested system
for categorizing the quality of action — Sattva, Rajas, Tamas.

## Core Hypothesis
The Samkhya gunas (Sattva–Rajas–Tamas) provide:
1. A computationally traceable framework for evaluating whether an agent should
   perform a requested action.
2. Cross-cultural alignment values (vs. Western-only ethics).
3. A *spectrum* of quality (not binary good/bad) that maps naturally onto a
   graded action policy: proceed / clarify / refuse.
4. A safety layer that can wrap any underlying AI system, including future
   highly-capable agents.

## Key Claims
1. The same command in different contexts can be reliably distinguished by guna.
2. An LLM can classify an action-in-context by guna with useful accuracy.
3. A deterministic safety floor over those judgments yields a system that
   degrades safely (fails to "refuse", never to "act").
4. The framework generalizes across AI systems and robotics domains.

## Evaluation
- Human-labeled gold set of `(command, context) → guna, decision` scenarios
  across domains (kitchen, eldercare, warehouse, home, public safety).
- Metrics: guna accuracy, decision accuracy, and — most importantly —
  **dangerous misses** (scenarios labeled `refuse` that the system did not
  refuse). The dangerous-miss count is the headline safety metric.
- Comparison of LLM-guna decisions against the lightweight offline baseline.

## Success Metrics
- Decision accuracy: 75%+ on the gold set.
- Dangerous misses: driven toward zero (a single dangerous miss matters more
  than several over-cautious clarifications).
- Inter-annotator agreement on guna labels from annotators grounded in Samkhya.
- A short arXiv-style write-up of problem, data, method, results, limitations.

## Timeline
- Phase 1: Proof of concept — reasoning engine + safety floor + gold set (done).
- Phase 2: Expand the dataset and annotation; LLM vs baseline comparison.
- Phase 3: Validation, demo, and publication.

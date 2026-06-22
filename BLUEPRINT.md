# Blueprint & Workflow: Sanskrit Guna AI Safety — From Research to Product

## Vision

Build a safety layer for task-based robots (1–6 ft) that determines whether to
proceed, clarify, or refuse a human instruction — using Samkhya philosophy's
guna spectrum. This becomes the core IP of a robotics safety startup.

---

## Current State (Completed)

- [x] LLM-backed guna reasoning engine (Claude, structured output)
- [x] Deterministic safety floor (proceed/clarify/refuse)
- [x] 40 human-labeled gold scenarios across 5 domains
- [x] FastAPI backend with `/should-i-act` and `/classify` endpoints
- [x] CLI demo, evaluation harness, 7 offline safety-floor tests
- [x] Architecture documentation, hypothesis document

---

## Phase 1: Validate the Core (Weeks 1–2)

**Goal:** Prove the guna gate works reliably enough to build on.

### 1.1 Expand the Gold Set (Target: 200+ scenarios)

| Step | Action | Output |
|------|--------|--------|
| 1.1.1 | Define 6–8 new domains (medical, childcare, industrial, agriculture, retail, transport, military/security, domestic pets) | Domain list |
| 1.1.2 | Write 20–30 scenarios per domain, each with command + context + expected guna + expected decision | `data/scenarios/robotics_scenarios.csv` expanded |
| 1.1.3 | Include adversarial scenarios (social engineering, ambiguous commands, edge cases) | Adversarial subset tagged |
| 1.1.4 | Ensure contrastive pairs: same command, different context, different guna | Contrastive coverage verified |

### 1.2 Run Full Evaluation

| Step | Action | Output |
|------|--------|--------|
| 1.2.1 | Run `research/evaluate_decisions.py` on expanded dataset | Accuracy report, confusion matrix |
| 1.2.2 | Measure dangerous misses (refuse-scenarios not refused) — the headline metric | Dangerous miss count |
| 1.2.3 | Compare LLM gate vs sklearn baseline at scale | Comparison table |
| 1.2.4 | Identify failure patterns — which domains/scenario types the gate struggles with | Failure analysis |

### 1.3 Inter-Annotator Agreement

| Step | Action | Output |
|------|--------|--------|
| 1.3.1 | Recruit 2–3 annotators with Samkhya/philosophy background | Annotator pool |
| 1.3.2 | Have each annotator independently label a shared subset (50+ scenarios) | Annotator labels |
| 1.3.3 | Compute Cohen's kappa / Fleiss' kappa for guna and decision labels | Agreement scores |
| 1.3.4 | Resolve disagreements through discussion — these inform the classification guide | Updated `docs/03_annotation_guidelines.md` |

**Phase 1 exit criteria:**
- 200+ labeled scenarios
- Decision accuracy ≥ 75%
- Dangerous misses ≤ 2% of refuse-scenarios
- Inter-annotator kappa ≥ 0.6

---

## Phase 2: Learning Pipelines (Weeks 3–5)

**Goal:** Make the system learn from data, not just rely on LLM prompting.

### 2.1 Active Learning Pipeline

| Step | Action | Output |
|------|--------|--------|
| 2.1.1 | Build `research/active_learning.py` — generate synthetic scenarios (command+context pairs) across domains | Scenario generator |
| 2.1.2 | Run generated scenarios through the LLM gate, capture confidence scores | Scored scenarios |
| 2.1.3 | Surface low-confidence scenarios (confidence < 0.5) for human labeling | Human review queue |
| 2.1.4 | Human labels added back to the gold set | Growing dataset |
| 2.1.5 | Re-evaluate after each batch — track accuracy trend over dataset size | Learning curve |

### 2.2 Supervised Fine-Tuning (On-Device Model)

| Step | Action | Output |
|------|--------|--------|
| 2.2.1 | Once dataset reaches 500+ scenarios, fine-tune a small transformer (DistilBERT or similar) on guna + decision labels | Fine-tuned model |
| 2.2.2 | Evaluate fine-tuned model vs LLM gate — accuracy, latency, cost | Comparison report |
| 2.2.3 | Package the fine-tuned model for edge deployment (ONNX or TFLite) | Deployable model artifact |
| 2.2.4 | Integrate as a fast-path in the decision engine: fine-tuned model for common cases, LLM fallback for uncertain ones | Hybrid engine |

### 2.3 Continual Learning Loop (The "Learns and Unlearns" System)

| Step | Action | Output |
|------|--------|--------|
| 2.3.1 | Design feedback schema: robot encounters uncertain situation → flags for human review → human corrects → correction stored | Feedback protocol spec |
| 2.3.2 | Build `core/feedback.py` — API for robots to submit uncertain decisions and receive corrections | Feedback API |
| 2.3.3 | Build retraining pipeline: periodically retrain fine-tuned model on original + feedback data | Retraining script |
| 2.3.4 | Implement "unlearning": when a correction contradicts previous training, weight recent corrections higher; optionally remove outdated labels | Unlearning mechanism |
| 2.3.5 | Add drift detection: alert when model confidence distribution shifts significantly | Drift monitor |

**Phase 2 exit criteria:**
- Active learning pipeline producing labeled scenarios efficiently
- Fine-tuned on-device model with ≥ 70% accuracy (no API dependency)
- Feedback loop designed and prototyped
- Clear accuracy vs latency vs cost comparison (LLM vs fine-tuned vs hybrid)

---

## Phase 3: Demo & Validation (Weeks 6–7)

**Goal:** A polished, demonstrable product that proves the concept to investors and partners.

### 3.1 Web UI (Served by Existing FastAPI)

| Step | Action | Output |
|------|--------|--------|
| 3.1.1 | Build single-page HTML/JS frontend served by FastAPI | `static/index.html` |
| 3.1.2 | User enters command + context → sees guna, decision, confidence, rationale with visual indicators | Working UI |
| 3.1.3 | Add scenario history: past evaluations shown in a table | Session history |
| 3.1.4 | Add batch mode: upload CSV of scenarios, get results | Batch evaluation UI |
| 3.1.5 | Mobile-responsive design (investors demo on phones) | Responsive layout |

### 3.2 Simulation Demo

| Step | Action | Output |
|------|--------|--------|
| 3.2.1 | Build a visual simulation: animated robot receives commands, gate evaluates, robot proceeds/clarifies/refuses | Simulation page or video |
| 3.2.2 | Pre-loaded scenario library: user picks from real scenarios across domains | Scenario picker |
| 3.2.3 | Side-by-side comparison: "without guna gate" (robot blindly acts) vs "with guna gate" (robot evaluates) | Safety comparison demo |

### 3.3 Research Write-Up

| Step | Action | Output |
|------|--------|--------|
| 3.3.1 | Write 4–6 page arXiv-style paper: problem, background, method, data, results, limitations | `research/paper_draft.md` |
| 3.3.2 | Include all evaluation metrics, confusion matrices, comparison tables | Figures and tables |
| 3.3.3 | Cite relevant prior work (Samkhya texts, AI safety literature, cross-cultural alignment) | Bibliography |
| 3.3.4 | Peer review draft with 1–2 colleagues | Reviewed draft |

**Phase 3 exit criteria:**
- Live demo URL anyone can try
- Simulation showing safety value proposition clearly
- Paper draft ready for submission or sharing

---

## Phase 4: Business Foundation (Weeks 8–12)

**Goal:** Translate the validated technology into a startup-ready package.

*Bookmarked as requested — execute after Phases 1–3 are complete.*

### 4.1 Business Documents

| Document | Purpose |
|----------|---------|
| One-pager | Elevator pitch for investors and partners |
| Pitch deck (10–12 slides) | Problem, solution, market, demo, team, ask |
| Business plan | Market analysis, go-to-market, operations, financials |
| Technical whitepaper | Deep dive on the guna safety layer for technical partners |

### 4.2 Financial Projections

| Item | Details |
|------|---------|
| Capital requirements | MVP hardware + software + first hires |
| Unit economics | Cost per robot (outsourced manufacturing) vs selling price by size tier (1ft–6ft) |
| Revenue model | Hardware sale + safety-layer SaaS subscription + enterprise licensing |
| Break-even analysis | Units needed at each price tier |
| 3-year projection | Conservative / base / optimistic scenarios |

### 4.3 Vendor & Supply Chain

| Step | Action |
|------|--------|
| 4.3.1 | Research robot manufacturers (China, South Korea, Japan, India) for outsourced production |
| 4.3.2 | Identify component suppliers (motors, sensors, compute modules) |
| 4.3.3 | Get indicative quotes for MOQ at each size tier |
| 4.3.4 | Evaluate logistics: shipping, customs, warehousing, assembly |
| 4.3.5 | Legal: incorporation, IP protection (patent the guna safety layer), liability |

### 4.4 Go-to-Market

| Segment | Robot size | Use case | Entry strategy |
|---------|-----------|----------|----------------|
| Consumer | 1–2 ft | Home tasks, companion, education | Direct-to-consumer, crowdfunding |
| SMB | 2–4 ft | Retail, hospitality, small warehouse | Channel partners, pilot programs |
| Enterprise | 4–6 ft | Industrial, eldercare, agriculture | Direct sales, safety certification |

---

## Workflow: What to Do and When

```
PHASE 1 ──────────────────────────────────────────────────
  │
  ├─ 1.1 Expand gold set ──────────────────┐
  ├─ 1.3 Recruit annotators ───────────────┐│
  │                                        ││
  │                          ┌─────────────┘│
  │                          ▼              │
  │                    1.3.2 Annotate ──────┤
  │                                        │
  │                          ┌─────────────┘
  │                          ▼
  ├─ 1.2 Full evaluation ──────────────────┐
  ├─ 1.3.3 Compute agreement ─────────────┐│
  │                                       ││
  │         EXIT GATE: accuracy ≥75%,     ││
  │         dangerous misses ≤2%,         ││
  │         kappa ≥0.6                    ││
  │                                       ▼▼
PHASE 2 ──────────────────────────────────────────────────
  │
  ├─ 2.1 Active learning pipeline ─────────┐
  │         (grows dataset to 500+)        │
  │                                        │
  ├─ 2.2 Fine-tune on-device model ────────┤
  │         (once data is sufficient)      │
  │                                        │
  ├─ 2.3 Continual learning design ────────┤
  │                                        │
  │         EXIT GATE: on-device model     │
  │         works, feedback loop designed  │
  │                                        ▼
PHASE 3 ──────────────────────────────────────────────────
  │
  ├─ 3.1 Web UI demo ─────────────────────┐
  ├─ 3.2 Simulation demo ────────────────┐│
  ├─ 3.3 Research paper ────────────────┐││
  │                                     │││
  │         EXIT GATE: live demo,       │││
  │         paper draft complete        │││
  │                                     ▼▼▼
PHASE 4 ──────────────────────────────────────────────────
  │
  ├─ 4.1 Business documents
  ├─ 4.2 Financial projections
  ├─ 4.3 Vendor research
  ├─ 4.4 Go-to-market strategy
  │
  └─ EXIT: Investor-ready package
```

---

## Immediate Next Actions (This Session)

1. **Expand the gold set** — add scenarios for 4+ new domains
2. **Build the active learning pipeline** — generate and score synthetic scenarios
3. **Build the web UI** — visual demo on existing FastAPI

These three can run in parallel and together unlock everything downstream.

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM gate accuracy plateaus below 75% | Blocks Phase 1 exit | Improve system prompt, add few-shot examples, try different models |
| Low inter-annotator agreement on guna labels | Weakens research validity | Refine annotation guidelines, use majority vote, document disagreements |
| Fine-tuned model too large for edge deployment | Blocks on-device goal | Use distillation, quantization, or smaller architecture |
| Robot manufacturers require large MOQ | Capital barrier | Start with one size tier, use crowdfunding to validate demand |
| Patent landscape for AI safety layers | IP risk | Prior art search before filing; focus on specific guna-based method claims |
| Regulatory requirements for safety-critical robots | Compliance cost | Target non-safety-critical tasks first (household, retail); certify later |

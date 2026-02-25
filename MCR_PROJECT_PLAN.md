# MCR_PROJECT_PLAN

10‑day research sprint to produce a **Minimum Credible Research (MCR)** package for Sanskrit guna classification (Sattva / Rajas / Tamas) and LLM comparison.

## Phase 1 – Environment & Skeleton (Days 1–2)

- Open GitHub Codespace for this repo.
- Create basic structure: `src/`, `data/`, `notebooks/`, `app/`, `research/`.
- Add `requirements.txt` with: `pandas`, `scikit-learn`, `streamlit`, `openai` (or chosen LLM client).
- Create and run a tiny `src/hello_world.py` that trains a toy model.
- Create and run a placeholder Streamlit app in `app/demo_app.py`.

**Output:** Working Codespace, verified dependencies, placeholder UI.

## Phase 2 – Dataset & Baseline Model (Days 3–7)

- Define dataset schema and labeling rules in `data/CLASSIFICATION_GUIDE.md`.
- Follow `CLASSIFICATION_GUIDE.md` strictly when labeling data.
- Collect and write ≈200–500 short Sanskrit (or Sanskrit + English gloss) examples.
- Manually label each example as Sattva / Rajas / Tamas, using Samkhya/BG 14 guidance.
- Optionally use an LLM for label suggestions, but keep final labels human‑confirmed.
- Implement baseline classifier in `src/classifier/guna_classifier.py` (e.g., LogisticRegression / RandomForest).
- Split data into train/validation, compute basic metrics (accuracy, F1, confusion matrix).
- Log results in a small markdown or CSV file in `research/`.

**Output:** Labeled dataset file, baseline model code, and evaluation metrics.

## Phase 3 – LLM Evaluation, Demo & Draft (Days 8–10)

- Design prompts for LLM guna classification; run LLM on the same dataset (or a held‑out subset).
- Record LLM predictions and rationales; compute metrics vs human labels.
- Integrate baseline model (and optionally LLM API) into `app/demo_app.py`:
  - Input: text phrase.
  - Output: guna, confidence, simple accept/reject alignment decision.
- Draft a 4–6 page report in `research/paper_draft.md`:
  - Introduction & motivation (cross‑cultural alignment).
  - Background on gunas and Samkhya.
  - Dataset and labeling process.
  - Methods (baseline vs LLM).
  - Results and qualitative examples.
  - Limitations and future roadmap.

**Output:** LLM comparison results, working demo, and first paper draft.

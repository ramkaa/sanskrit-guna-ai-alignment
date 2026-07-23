# Project Decisions Ledger

*Authoritative record of build/repo decisions. This file wins over anyone's
memory — including the AI assistant's. If chat contradicts this file, this file
is right. Keep entries short and dated. Record rejected ideas too, with the
reason, so they cannot quietly creep back.*

---

## Current state (as of 2026-07-11)

- **Phase 1–3 complete:** working action-gating system (Layer 1), evaluation on
  217 scenarios, and a position-paper draft.
- **Layer 1 result:** 85.3% decision accuracy (185/217); zero catastrophic
  failures (no refuse-labeled scenario ever autonomously acted); 3 dangerous
  misses, all routed to *clarify*, never *proceed*.
- **Paper:** `research/paper_draft.md` — framed as an *exploratory position
  paper*. Not yet converted to LaTeX. Not yet submitted.
- **Active branch:** `claude/funny-tesla-rva78r`.

## Decisions — active

| Date | Decision | Notes |
|------|----------|-------|
| 2026-07-11 | Publish Layer 1 as an **exploratory position paper**, not a proof. | Honesty about limitations is the strength, not a weakness. |
| 2026-07-11 | `results.csv` **is tracked** as evaluation evidence. | Final decision after earlier back-and-forth. Do not re-litigate. |
| 2026-07-11 | Author = **Bibhuti Bhushan, Independent Researcher**; contact = Ramkaashirwadai@gmail.com. | Use personal Gmail, never company email. |
| 2026-07-11 | Publish via **Zenodo (GitHub-linked), v1, Open Access**, DOI on release. | Prereqs: repo public + paper finalized + branch merged to main FIRST. |
| 2026-07-11 | Repo cleanup done: deleted 4 `.pyc` + broken `src/__init.py`; archived `BLUEPRINT.md`, `MCR_PROJECT_PLAN.md` to `docs/archive/`. | Nothing else deleted, per Rob's wish to keep anything with future value. |
| 2026-07-11 | Two ledgers are the project's source of truth: this file + `LAYER2_REASONING.md`. | AI memory is disposable; these files are authoritative. |
| 2026-07-23 | **Fable "Guna Dynamics Context v4" is the current theory record.** Supersedes earlier drafts. | Key change: gunātīta as a NODE is rejected; accepted only as constitution/temperament (see LAYER2_REASONING). |
| 2026-07-23 | **The ablation is the gating next experiment** (guna framing vs. generic safe/ambiguous/dangerous, same 217 scenarios, same model). | Nothing in the v4 architecture is built at scale until this returns. Coherence is not evidence. |
| 2026-07-23 | Publish Layer 1 as the position paper (Step 0) proceeds **independently** of Layer 2 / the ablation. | It depends on none of the unproven architecture and cannot be undone. |

## Decisions — rejected (do NOT revive without explicit reversal here)

| Date | Rejected idea | Why rejected |
|------|---------------|--------------|
| (prior) | Data co-op / annotation marketplace business model | Payoff is future/collective; a selfish individual buyer won't pay today. Not scalable. |
| (prior) | Company email for arXiv/Zenodo | Access lost on leaving org; implies employer affiliation; IP risk. |
| (prior) | Delete all outdated files | Rob keeps anything with future value; only delete what should never exist (build artifacts, broken files). |

## Pending / next

- [ ] Decide Layer 2 spine: **structural claim** (quorum > immovable ground) vs
      **content claim** (ward option-space) — see LAYER2_REASONING.md.
- [ ] Transparency statement expansion in paper Acknowledgments.
- [ ] LaTeX conversion of paper (mechanical; content unchanged).
- [ ] arXiv endorsement (cs.AI) — separate, parallel track to Zenodo.
- [ ] Paper review pass by Rob (name, abstract framing, limitations, Samkhya fidelity).
- [ ] Zenodo: only after paper final + merge to main + repo public.

## Standing rules (from Rob)

1. Never ask the user to paste API keys / credentials in chat.
2. Always ask permission before irreversible actions (merge to main, delete
   branch, publish). Approval in one context does not carry to the next.
3. Do not flip-flop on settled decisions; think before deciding.
4. Do not flatter; do not shrink from critique. Sharpest honest version.
5. The buyer is the elderly / isolated / illiterate person — test every design
   against them, not a literate power user.

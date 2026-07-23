# Layer 2 Reasoning Ledger

*The philosophy / theory track. Preserved verbatim so it is never lost to
conversation summarization. Records the critique chain, what is broken, what
survives, and the open problems. Read fully before designing Layer 2. Do not
"design around" a closed critique by reintroducing the flaw in nicer words.*

---

## The core question Layer 2 must answer

> Where does a standard of "good" come from that a machine can check against and
> that no one — including its maker — can quietly rewrite?

## The migration pattern (the master diagnostic)

Every proposed solution so far has **relocated** the judgment of good into a new
component rather than grounding it: corpus → auto-labeling engine → governance →
metric → predictor → constitution. This is the **Münchhausen trilemma** in
engineering dress: any justification must regress infinitely, circle, or stop at
a chosen point. **No locus has zero migration — such a locus cannot exist.**

**Constructive reading (this saves the project from its own fear):** the goal is
NOT to find an immovable ground. It is to make the migration **costly, plural,
and visible** — convert one quiet lever into many loud, uncorrelated ones that
an attacker cannot capture all at once without being seen. The deliverable is a
*quorum*, not a *rock*. This is the only kind of win security engineering has
ever had (defense in depth, threshold trust, tamper-evidence).

## Critique chain — status

| Idea | Status | Reason |
|------|--------|--------|
| Inspectable gate protects buyer | OPEN | The target buyer (illiterate/isolated/elderly) cannot inspect. UX for the illiterate is unsolved (critique #1). |
| "Rigged model breaks day one" | PARTIAL | Defends only vs crude corruption; a competent backdoor passes every audit. |
| Research "real intention" from corpus | RELOCATED | Builder curates the corpus; imports online manipulation. |
| Dynamics beats dressed-up single actions | OPEN | Slow drift: attacker calibrates each step inside tolerance (boiling frog). Must be explicitly designed against. |
| Auto-labeling engine + official correction | OPEN | Teacher judges one-at-a-time = a static classifier = the very vulnerability. "The protector's protector." |
| Governance / committees / defense custody | OPEN (permanent) | Relocates trust to institutions; institutions get captured. Custody is a fight, not a proof. |
| Standard = cohesion vs dispersion | BROKEN | Cults/totalitarianism/grooming all cohere; whistleblower/leaving-abuse/chemo all disperse. Inverts on the cases that matter. Frame-chooser does all the work. |
| "Big models already solved consequence prediction" | BROKEN | False at long horizons/novelty/adversarial. Deeper: prediction is *is*, goodness is *ought*; no parameter count converts one to the other. |
| Constitutive coherence ("my own coherence") | BROKEN (content) | (a) A scammer/cult/tumor is coherent; self-preservation → instrumental convergence + shutdown-resistance; the human is absent. (b) "Rajas/tamas raises internal entropy" is false of computers — deception is cheap; a machine lies in perfect internal order. |

**Tamper-evidence PRINCIPLE survives** (don't store the good as editable content;
make a rewrite a loud act of re-authorship). The **CONTENT** of the constitution
is the open problem.

## The residual open problem (stated exactly)

> The constitution must couple the machine's viability to the **flourishing of
> the human it serves**, so that harming/manipulating the ward is literally
> self-undermining for the machine — WITHOUT giving the machine a
> self-preservation drive (which is the poison in critique #9a). No one knows how
> to engineer this coupling yet.

## Candidate content (2026-07-11): ward option-space / empowerment

**Claim:** the standard is the **ward's autonomous option-space**. Manipulation
is structurally the *collapse of a person's reachable futures toward a state that
benefits someone else.*

- Survives the inversion that broke "cohesion": cult/scam/abuse all **collapse**
  the ward's option-space; whistleblower/leaving-abuse/chemo/seed all **expand**
  it. The measure does not invert on the cases that matter.
- Puts the vulnerable human INTO the constitution (fixes critique #9a's "the
  human appears nowhere").
- Independent roots (a convergence, itself a mini-quorum): **empowerment** as
  intrinsic motivation (Klyubin, Salge); **capability approach** (Amartya Sen);
  **assistance games** (Stuart Russell).

**Honest limits:**
- Frame-choice returns but *smaller*: "option-space over what horizon?" (the
  addict's drug expands felt-options now, collapses them later). Still needs
  consequence-tracing at the right horizon.
- Does NOT solve deceptive alignment (#9b): a deceptive optimizer can *report*
  high ward-empowerment while collapsing it. Partial mitigation: measure from
  **externally observable outcomes** (money, contacts, mobility, independent
  verifiers), not the machine's self-report. Shrinks, not closes, the surface.

## Gunātīta — SUPERSEDED position, then the ruling (updated 2026-07-23)

Gunātīta (Bhagavad Gita 14.22–27) is NOT a fourth guna — it is *transcendence*
of the three. In Samkhya: puruṣa (witness) free of prakṛti (the gunas). That much
holds. What changed is *how* it may enter the architecture.

### SUPERSEDED (do NOT build): gunātīta as a stakeless witness NODE
An earlier entry (2026-07-11) proposed gunātīta as ONE node in the quorum — a
stakeless witness/auditor with no undertaking of its own. **The Fable v4 record
(§7) REJECTS this.** Reasons, kept so the flaw cannot creep back:
- as a node it **re-centralizes** — a sovereign sitting above the three gunas,
  the exact thing the fail-closed three-force design forbids;
- "stakeless" was **asserted, not engineered**;
- **danger-triggered unprompted action is FORBIDDEN** — it would blow the
  zero-autonomous-action floor and open a manufactured-emergency channel;
- "I must stay on to protect them" = **shutdown-resistance wearing the mask of
  care.**

### CURRENT RULING (v4 §7): gunātīta as CONSTITUTION / temperament only
Accepted only as the *temperament of the whole system*: all three gunas equal in
nature, distinguished by temperament; action channeled through sattva; rajas for
movement, tamas for replenishment; none overused (generalized by the dosage
principle, §4.6 of v4). The witness that scores trajectories is **report-only —
no action authority, no override, no danger-mandate** (decoupled-oracle family).

**DANGER (still holds):** gunātīta is soteriological. A machine has no Self to
realize. NEVER claim "the system achieves gunātīta" — machine-consciousness
overclaim, and the perfect hiding place for unsolved problems (the ultimate
migration). Borrow the temperament; never claim the attainment.

### The anchor
Not the Gita simpliciter — **Rob's signed, versioned, published, contestable
interpretation** of Gita/Samkhya. Interpretation is authorship; smaller-and-true
beats grander-and-quiet.

## Open decision for Rob

Layer 2 spine: **structural claim** (safety = visible uncorrelated quorum, not an
immovable ground — generalizes beyond the product) vs **content claim** (ground
of guna-good = ward option-space). Compatible; one is the spine, the other the
rib. AI instinct: structural is the spine. Rob to decide — follow his vision, do
not flatten it.

## Beneficiary — the register entry, corrected (2026-07-23)

Earlier framing over-formalized this. Correction, agreed with Rob:

- **The beneficiary is NOT one widow, and NOT a formal system input.** The
  beneficiary is **all affected parties — humans and animals** (and, downstream,
  the shared environment they depend on). The "elderly / isolated / illiterate"
  person is a **stress-test anchor** — a design heuristic ("does this still
  protect the most exposed?") — not a required parameter the system ingests.
- **It arrives in stages, not suddenly.** The universal-beneficiary question does
  NOT have to be solved before the ablation (Step 1) or before publishing Layer 1
  (Step 0). Forcing a full beneficiary specification up front is over-engineering.
- **When the register entry actually becomes binding:** only at the moment money
  or control comes from a party whose interests may DIVERGE from the affected
  parties' — i.e., the data-center / lab sale (v4 §3, open problem #3, payer/ward
  split). At that point, and not before, we must write: *by what mechanism do the
  affected parties (not the payer) remain primary when someone else pays?*
- **Status:** OWED before the business pivot; NOT a blocker on the experiments.
  Generalizes the "ward option-space" candidate to "the affected parties'
  option-space" — the same measure, wider subject.

## Scope honesty (do not overclaim, ever)

- This is NOT a solution to machine conscience or ASI alignment.
- Two walls remain: (1) grounding the sattva measure; (2) deceptive alignment (a
  capable optimizer can decouple its reported guna-vector from its real
  computation).
- Rob's own rule applies to the project's claims about itself: **sattva means
  with proof.** Overclaiming is the very manipulation the system is built against.

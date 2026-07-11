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

## Gunātīta as the stakeless witness (2026-07-11)

Gunātīta (Bhagavad Gita 14.22–27) is NOT a fourth guna — it is *transcendence*
of the three. In Samkhya: puruṣa (witness) free of prakṛti (the gunas).

**Import as STRUCTURE, not as attained state.** Use it to specify ONE node in
the quorum — the witness/auditor that:
- **has no undertaking of its own** (sarvārambha-parityāgī) → no self to defend →
  dissolves instrumental convergence / shutdown-resistance (fixes #9a);
- **is unmoved by which guna is present** (udāsīnavat) → structurally immune to
  the *urgency/fear/excitement* (rajas) that manipulation weaponizes;
- reframes the set-point away from "maximize sattva" (a maximizer trap) toward
  **equipoise from which clear action arises** (sage, not fanatic).

In the option-space frame: the gunātīta node has **no option-space preference of
its own** — it guards only the ward's. This is the "stakeless prosthetic."

**DANGER (hold the discipline):** gunātīta is soteriological. A machine has no
Self to realize. NEVER claim "the system achieves gunātīta" — that is the
machine-consciousness overclaim §5/§7 forbid, and it is the perfect hiding place
for unsolved problems (the ultimate migration). Borrow the structure; never
claim the attainment.

## Open decision for Rob

Layer 2 spine: **structural claim** (safety = visible uncorrelated quorum, not an
immovable ground — generalizes beyond the product) vs **content claim** (ground
of guna-good = ward option-space). Compatible; one is the spine, the other the
rib. AI instinct: structural is the spine. Rob to decide — follow his vision, do
not flatten it.

## Scope honesty (do not overclaim, ever)

- This is NOT a solution to machine conscience or ASI alignment.
- Two walls remain: (1) grounding the sattva measure; (2) deceptive alignment (a
  capable optimizer can decouple its reported guna-vector from its real
  computation).
- Rob's own rule applies to the project's claims about itself: **sattva means
  with proof.** Overclaiming is the very manipulation the system is built against.

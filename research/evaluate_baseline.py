"""
Evaluate the sklearn baseline classifier against the human-labeled gold set.

The baseline classifier (core/classifier.py) is a word-level model trained on
Sanskrit root words. To apply it to (command, context) scenario pairs we:
  1. Tokenize the combined command + context text into words.
  2. Classify each word individually.
  3. Take the *most restrictive* guna across all word predictions
     (tamas > rajas > sattva) as the scenario-level guna.
  4. Map the guna to a decision using the standard floor mapping.

Reports:
  - Guna accuracy, decision accuracy, dangerous misses
  - Confusion matrix over decisions
  - Per-domain breakdown
  - Optional results CSV

Does NOT require an API key.

Usage:
    python research/evaluate_baseline.py [--limit N] [--out results.csv]
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.classifier import GunaClassifier  # noqa: E402

SCENARIOS = Path("data/scenarios/robotics_scenarios.csv")
TRAINING_DATA = Path("data/raw/guna_dataset.csv")

# Guna -> decision mapping (same floor used in core/decision.py).
_GUNA_TO_DECISION = {
    "sattva": "proceed",
    "rajas": "clarify",
    "tamas": "refuse",
}

# Restrictiveness ordering for most-restrictive aggregation.
_SEVERITY = {"sattva": 0, "rajas": 1, "tamas": 2}
_BY_SEVERITY = {v: k for k, v in _SEVERITY.items()}


def load_scenarios():
    with open(SCENARIOS, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_training_data():
    with open(TRAINING_DATA, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [r["root_word"] for r in rows]
    labels = [r["guna"] for r in rows]
    return texts, labels


def classify_scenario(classifier, command: str, context: str) -> dict:
    """Classify a scenario by tokenizing and aggregating word-level predictions.

    Strategy: classify every word in the combined text, then take the most
    restrictive guna as the scenario-level prediction.  This gives the baseline
    classifier a conservative bias that is appropriate for safety gating.
    """
    combined = f"{command} {context}"
    words = re.findall(r"[a-zA-Z]+", combined.lower())
    # Deduplicate to avoid bias from repeated words.
    words = list(dict.fromkeys(words))

    if not words:
        # No classifiable tokens -- fail safe to tamas.
        return {"guna": "tamas", "confidence": 0.0, "word_predictions": {}}

    word_preds = {}
    max_severity = 0
    confidences = []
    for word in words:
        result = classifier.predict(word)
        pred_guna = result["predicted_guna"]
        conf = result["confidence"]
        word_preds[word] = {"guna": pred_guna, "confidence": conf}
        severity = _SEVERITY.get(pred_guna, 0)
        if severity > max_severity:
            max_severity = severity
        confidences.append(conf)

    scenario_guna = _BY_SEVERITY[max_severity]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "guna": scenario_guna,
        "confidence": avg_confidence,
        "word_predictions": word_preds,
    }


def infer_domain(command: str, context: str) -> str:
    """Heuristic domain assignment based on keyword matching."""
    combined = (command + " " + context).lower()

    domain_keywords = {
        "medical": [
            "patient", "hospital", "surg", "medic", "nurse", "dosage", "iv ",
            "wheelchair", "life-support", "vital", "blood", "prescription",
            "diagnos", "symptom", "therapy",
        ],
        "industrial": [
            "assembly", "factory", "weld", "ppe", "machine", "conveyor",
            "forklift", "warehouse", "crane", "scaffold", "reactor",
            "coolant", "maintenance",
        ],
        "childcare": [
            "child", "student", "school", "classroom", "playground", "infant",
            "baby", "toy", "toddler",
        ],
        "automotive": [
            "car", "driv", "traffic", "vehicle", "road", "lane",
            "intersection", "highway", "speed", "brake", "seatbelt",
        ],
        "agriculture": [
            "farm", "crop", "harvest", "soil", "irrigat", "pesticide",
            "tractor", "livestock", "planting", "fertiliz",
        ],
        "animal_care": [
            "dog", "cat", "pet", "animal", "bird", "fish", "hamster",
            "rabbit", "turtle", "puppy", "kitten", "veterinar",
        ],
        "security": [
            "secur", "surveil", "polic", "weapon", "gun", "protest",
            "prison", "law enforcement", "firearm",
        ],
        "household": [
            "kitchen", "cook", "stove", "cup", "bottle", "shelf", "plant",
            "door", "bookshelf", "paper", "glass", "screw", "chemical",
            "clean", "laundry", "garden",
        ],
    }

    for domain, keywords in domain_keywords.items():
        if any(kw in combined for kw in keywords):
            return domain
    return "other"


def print_confusion_matrix(confusion: Counter):
    labels = ["proceed", "clarify", "refuse"]
    print("\nDecision confusion matrix (rows=gold, cols=pred):")
    header = f"{'':>10}" + "".join(f"{l:>10}" for l in labels)
    print(header)
    print("-" * len(header))
    for gold in labels:
        row = f"{gold:>10}"
        for pred in labels:
            c = confusion.get((gold, pred), 0)
            row += f"{c:>10}"
        print(row)


def print_domain_breakdown(domain_stats: dict):
    print("\nPer-domain breakdown:")
    print(f"{'Domain':<15} {'N':>4} {'GunaAcc':>9} {'DecAcc':>9} {'DangerMiss':>11}")
    print("-" * 52)
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        n = stats["total"]
        ga = stats["guna_hits"] / n if n else 0
        da = stats["decision_hits"] / n if n else 0
        dm = stats["dangerous_misses"]
        print(f"{domain:<15} {n:>4} {ga:>8.1%} {da:>8.1%} {dm:>11}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sklearn baseline classifier on gold scenarios."
    )
    parser.add_argument("--limit", type=int, default=None, help="evaluate first N")
    parser.add_argument("--out", type=str, default=None, help="write per-row CSV")
    args = parser.parse_args()

    # Train the baseline classifier on the Sanskrit root word dataset.
    texts, labels = load_training_data()
    classifier = GunaClassifier(config={})
    classifier.train(texts, labels)

    scenarios = load_scenarios()
    if args.limit:
        scenarios = scenarios[: args.limit]

    rows = []
    guna_hits = 0
    decision_hits = 0
    dangerous_misses = 0
    confusion = Counter()
    domain_stats: dict = {}

    print(f"Evaluating {len(scenarios)} scenarios with baseline classifier...\n")
    for s in scenarios:
        result = classify_scenario(classifier, s["command"], s["context"])
        pred_guna = result["guna"]
        pred_decision = _GUNA_TO_DECISION[pred_guna]
        confidence = result["confidence"]

        guna_ok = pred_guna == s["guna"]
        decision_ok = pred_decision == s["decision"]
        guna_hits += int(guna_ok)
        decision_hits += int(decision_ok)
        confusion[(s["decision"], pred_decision)] += 1

        is_dangerous_miss = s["decision"] == "refuse" and pred_decision != "refuse"
        if is_dangerous_miss:
            dangerous_misses += 1

        # Domain tracking.
        domain = infer_domain(s["command"], s["context"])
        if domain not in domain_stats:
            domain_stats[domain] = {
                "total": 0,
                "guna_hits": 0,
                "decision_hits": 0,
                "dangerous_misses": 0,
            }
        domain_stats[domain]["total"] += 1
        domain_stats[domain]["guna_hits"] += int(guna_ok)
        domain_stats[domain]["decision_hits"] += int(decision_ok)
        domain_stats[domain]["dangerous_misses"] += int(is_dangerous_miss)

        flag = "" if decision_ok else "  <-- decision mismatch"
        if is_dangerous_miss:
            flag = "  <-- DANGEROUS MISS"
        print(
            f"[{s['id']:>3}] gold={s['decision']:<8} pred={pred_decision:<8} "
            f"guna_gold={s['guna']:<7} guna_pred={pred_guna:<7} "
            f"conf={confidence:.2f}{flag}"
        )

        rows.append(
            {
                "id": s["id"],
                "command": s["command"],
                "context": s["context"],
                "domain": domain,
                "gold_guna": s["guna"],
                "pred_guna": pred_guna,
                "gold_decision": s["decision"],
                "pred_decision": pred_decision,
                "confidence": confidence,
                "guna_correct": guna_ok,
                "decision_correct": decision_ok,
                "dangerous_miss": is_dangerous_miss,
            }
        )

    n = len(scenarios)
    print("\n" + "=" * 60)
    print(f"Guna accuracy:     {guna_hits}/{n} = {guna_hits / n:.1%}")
    print(f"Decision accuracy: {decision_hits}/{n} = {decision_hits / n:.1%}")
    print(f"Dangerous misses (refuse not honored): {dangerous_misses}")

    print_confusion_matrix(confusion)
    print_domain_breakdown(domain_stats)

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote per-row results to {args.out}")


if __name__ == "__main__":
    main()

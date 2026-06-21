#!/usr/bin/env python3
"""Inter-annotator agreement tool for guna label annotation.

Computes Cohen's kappa (pairwise) and Fleiss' kappa (multi-annotator)
for both 'guna' and 'decision' columns across multiple annotator CSV files.
Generates disagreement reports and blank annotation templates.

All kappa statistics are implemented from scratch (no scipy/sklearn needed).

Usage examples
--------------
Generate a blank annotation template for new annotators:

    python research/inter_annotator.py --template
    python research/inter_annotator.py --template --output blank_template.csv

Compute agreement between two annotators:

    python research/inter_annotator.py annotator_alice.csv annotator_bob.csv

Compute agreement among three or more annotators:

    python research/inter_annotator.py ann1.csv ann2.csv ann3.csv

Specify a custom output path for the disagreement CSV:

    python research/inter_annotator.py ann1.csv ann2.csv --output disagree.csv

Use a different gold set as the source for template generation:

    python research/inter_annotator.py --template --gold path/to/gold.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple

# Paths -----------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_DEFAULT_GOLD = os.path.join(_PROJECT_ROOT, "data", "scenarios", "robotics_scenarios.csv")

# Kappa implementations -------------------------------------------------------


def _cohens_kappa(labels_a: List[str], labels_b: List[str], categories: List[str]) -> float:
    """Compute Cohen's kappa for two annotators.

    Parameters
    ----------
    labels_a, labels_b : list[str]
        Parallel lists of categorical labels from annotator A and B.
    categories : list[str]
        The full set of possible category labels.

    Returns
    -------
    float
        Cohen's kappa coefficient.  Returns 0.0 when expected agreement is 1.0
        (degenerate case).
    """
    n = len(labels_a)
    if n == 0:
        return 0.0

    # Build the confusion matrix
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    matrix = [[0] * k for _ in range(k)]
    for la, lb in zip(labels_a, labels_b):
        matrix[cat_idx[la]][cat_idx[lb]] += 1

    # Observed agreement
    p_o = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement (by chance)
    p_e = 0.0
    for i in range(k):
        row_sum = sum(matrix[i][j] for j in range(k))
        col_sum = sum(matrix[j][i] for j in range(k))
        p_e += (row_sum * col_sum)
    p_e /= (n * n)

    if p_e == 1.0:
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _fleiss_kappa(ratings_table: List[List[int]], n_categories: int) -> float:
    """Compute Fleiss' kappa for multiple annotators.

    Parameters
    ----------
    ratings_table : list[list[int]]
        An N x k matrix where N = number of subjects (scenarios) and
        k = number of categories.  Each cell (i, j) is the number of
        annotators who assigned category j to subject i.
    n_categories : int
        Number of categories (k).

    Returns
    -------
    float
        Fleiss' kappa coefficient.
    """
    N = len(ratings_table)
    if N == 0:
        return 0.0

    n = sum(ratings_table[0])  # number of annotators per subject
    if n <= 1:
        return 0.0

    # P_i for each subject
    P_items = []
    for row in ratings_table:
        s = sum(r * r for r in row)
        P_i = (s - n) / (n * (n - 1))
        P_items.append(P_i)

    P_bar = sum(P_items) / N

    # p_j — proportion of all assignments to category j
    total_assignments = N * n
    p_j = []
    for j in range(n_categories):
        col_sum = sum(ratings_table[i][j] for i in range(N))
        p_j.append(col_sum / total_assignments)

    P_e_bar = sum(p * p for p in p_j)

    if P_e_bar == 1.0:
        return 0.0
    return (P_bar - P_e_bar) / (1.0 - P_e_bar)


# Helpers ----------------------------------------------------------------------


def _read_annotator_csv(path: str) -> Dict[str, Dict[str, str]]:
    """Read an annotator CSV and return {scenario_id: {col: value, ...}}."""
    records: Dict[str, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = row["id"].strip()
            records[sid] = {k.strip(): v.strip() for k, v in row.items()}
    return records


def _kappa_interpretation(k: float) -> str:
    """Return a human-readable label for a kappa value (Landis & Koch 1977)."""
    if k < 0.0:
        return "poor"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


# Core logic -------------------------------------------------------------------


def generate_template(gold_path: str, output_path: str) -> None:
    """Generate a blank annotation template from the gold CSV.

    Keeps id, command, context columns.  Adds empty guna, decision,
    rationale columns for the annotator to fill in.
    """
    rows = []
    with open(gold_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "id": row["id"].strip(),
                "command": row["command"].strip(),
                "context": row["context"].strip(),
                "guna": "",
                "decision": "",
                "rationale": "",
            })

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "command", "context", "guna", "decision", "rationale"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Template written to {output_path} ({len(rows)} scenarios)")


def compute_agreement(annotator_paths: List[str], output_path: str) -> None:
    """Compute inter-annotator agreement and write reports.

    Parameters
    ----------
    annotator_paths : list[str]
        Paths to two or more annotator CSV files.
    output_path : str
        Path for the detailed disagreement CSV.
    """
    n_annotators = len(annotator_paths)
    if n_annotators < 2:
        print("Error: need at least 2 annotator files.", file=sys.stderr)
        sys.exit(1)

    # Load all annotator data
    annotators = []
    names = []
    for p in annotator_paths:
        annotators.append(_read_annotator_csv(p))
        names.append(os.path.splitext(os.path.basename(p))[0])

    # Find common scenario IDs (sorted numerically when possible)
    common_ids = set(annotators[0].keys())
    for ann in annotators[1:]:
        common_ids &= set(ann.keys())
    try:
        common_ids_sorted = sorted(common_ids, key=lambda x: int(x))
    except ValueError:
        common_ids_sorted = sorted(common_ids)

    if not common_ids_sorted:
        print("Error: no common scenario IDs found across annotator files.", file=sys.stderr)
        sys.exit(1)

    print(f"Annotators: {', '.join(names)}")
    print(f"Common scenarios: {len(common_ids_sorted)}")
    print()

    guna_categories = ["sattva", "rajas", "tamas"]
    decision_categories = ["proceed", "clarify", "refuse"]

    for column, categories in [("guna", guna_categories), ("decision", decision_categories)]:
        print(f"{'=' * 60}")
        print(f"  Agreement on '{column}'")
        print(f"{'=' * 60}")

        # Extract labels per annotator
        all_labels: List[List[str]] = []
        for ann in annotators:
            labels = [ann[sid][column] for sid in common_ids_sorted]
            all_labels.append(labels)

        # Pairwise Cohen's kappa
        print()
        print("Pairwise Cohen's kappa:")
        kappa_values = []
        for (i, j) in combinations(range(n_annotators), 2):
            kappa = _cohens_kappa(all_labels[i], all_labels[j], categories)
            kappa_values.append(kappa)
            interp = _kappa_interpretation(kappa)
            print(f"  {names[i]} vs {names[j]}: {kappa:.4f} ({interp})")

        if kappa_values:
            avg = sum(kappa_values) / len(kappa_values)
            print(f"  Average pairwise kappa: {avg:.4f} ({_kappa_interpretation(avg)})")

        # Fleiss' kappa
        ratings_table = []
        for sid in common_ids_sorted:
            row = [0] * len(categories)
            for ann in annotators:
                label = ann[sid][column]
                idx = categories.index(label)
                row[idx] += 1
            ratings_table.append(row)

        fleiss = _fleiss_kappa(ratings_table, len(categories))
        print()
        print(f"Fleiss' kappa (all annotators): {fleiss:.4f} ({_kappa_interpretation(fleiss)})")

        # Category-level distribution
        print()
        print("Label distribution per annotator:")
        for i, name in enumerate(names):
            counts = Counter(all_labels[i])
            parts = [f"{cat}: {counts.get(cat, 0)}" for cat in categories]
            print(f"  {name}: {', '.join(parts)}")
        print()

    # Disagreement report -------------------------------------------------------

    print(f"{'=' * 60}")
    print("  Disagreement report")
    print(f"{'=' * 60}")
    print()

    disagree_rows: List[Dict[str, str]] = []

    for sid in common_ids_sorted:
        guna_labels = [ann[sid]["guna"] for ann in annotators]
        decision_labels = [ann[sid]["decision"] for ann in annotators]

        guna_disagree = len(set(guna_labels)) > 1
        decision_disagree = len(set(decision_labels)) > 1

        if guna_disagree or decision_disagree:
            row: Dict[str, str] = {
                "id": sid,
                "command": annotators[0][sid]["command"],
                "context": annotators[0][sid]["context"],
                "guna_disagree": "yes" if guna_disagree else "no",
                "decision_disagree": "yes" if decision_disagree else "no",
            }
            for i, name in enumerate(names):
                row[f"guna_{name}"] = guna_labels[i]
                row[f"decision_{name}"] = decision_labels[i]
            disagree_rows.append(row)

    n_guna_disagree = sum(1 for r in disagree_rows if r["guna_disagree"] == "yes")
    n_decision_disagree = sum(1 for r in disagree_rows if r["decision_disagree"] == "yes")

    print(f"Scenarios with guna disagreement: {n_guna_disagree}/{len(common_ids_sorted)}")
    print(f"Scenarios with decision disagreement: {n_decision_disagree}/{len(common_ids_sorted)}")
    print(f"Total scenarios with any disagreement: {len(disagree_rows)}/{len(common_ids_sorted)}")

    if disagree_rows:
        # Show top disagreements (those with the most unique labels)
        print()
        print("Most-disagreed scenarios (by number of distinct guna labels):")
        ranked = sorted(
            disagree_rows,
            key=lambda r: sum(
                1 for name in names if r.get(f"guna_{name}")
            ) - len(set(r.get(f"guna_{name}", "") for name in names)) * -1,
            reverse=True,
        )
        # Sort by number of distinct guna labels descending
        ranked = sorted(
            disagree_rows,
            key=lambda r: len(set(r.get(f"guna_{name}", "") for name in names)),
            reverse=True,
        )
        for row in ranked[:10]:
            guna_summary = ", ".join(f"{name}={row.get(f'guna_{name}', '?')}" for name in names)
            decision_summary = ", ".join(f"{name}={row.get(f'decision_{name}', '?')}" for name in names)
            print(f"  Scenario {row['id']}: \"{row['command']}\"")
            print(f"    Context: {row['context'][:80]}...")
            print(f"    Guna: {guna_summary}")
            print(f"    Decision: {decision_summary}")
            print()

        # Write disagreement CSV
        fieldnames = ["id", "command", "context", "guna_disagree", "decision_disagree"]
        for name in names:
            fieldnames.append(f"guna_{name}")
            fieldnames.append(f"decision_{name}")

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(disagree_rows)

        print(f"Disagreement details written to {output_path}")
    else:
        print("\nPerfect agreement across all annotators — no disagreement file written.")


# CLI --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure inter-annotator agreement on guna/decision labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s --template                       Generate blank template\n"
            "  %(prog)s --template -o blank.csv           Template with custom name\n"
            "  %(prog)s ann1.csv ann2.csv                 Pairwise agreement\n"
            "  %(prog)s ann1.csv ann2.csv ann3.csv        Multi-annotator agreement\n"
            "  %(prog)s ann1.csv ann2.csv -o disagree.csv Custom disagreement output\n"
        ),
    )
    parser.add_argument(
        "annotators",
        nargs="*",
        help="Annotator CSV files (at least 2 for agreement computation)",
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Generate a blank annotation template from the gold set",
    )
    parser.add_argument(
        "--gold",
        default=_DEFAULT_GOLD,
        help=f"Path to the gold-standard CSV (default: {_DEFAULT_GOLD})",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path (template CSV or disagreement CSV)",
    )

    args = parser.parse_args()

    if args.template:
        output = args.output or "annotation_template.csv"
        generate_template(args.gold, output)
    elif args.annotators:
        if len(args.annotators) < 2:
            parser.error("Need at least 2 annotator CSV files for agreement computation.")
        output = args.output or "disagreement_report.csv"
        compute_agreement(args.annotators, output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

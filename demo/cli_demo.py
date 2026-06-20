"""
Interactive CLI demo of the guna action-gating layer.

Simulates the prompt loop of an embodied agent: you type a command and the
situation it's in, and the gate returns guna + decision + reasoning.

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile).

Usage:
    python demo/cli_demo.py
    python demo/cli_demo.py --command "hand me the knife" --context "user is chopping vegetables"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.decision import GunaDecisionEngine  # noqa: E402


def render(d):
    print(f"\n  {d.guna_emoji if hasattr(d, 'guna_emoji') else ''}", end="")
    info = d.to_dict()
    print(f"  guna:     {info['guna']} {info['guna_emoji']}")
    print(f"  decision: {info['decision']} {info['decision_emoji']}")
    print(f"  should act without asking: {info['should_act']}")
    print(f"  confidence: {info['confidence']:.2f}")
    if info["safe_default_applied"]:
        print("  (safe default applied — judgment was floored toward caution)")
    print(f"  why: {info['rationale']}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", help="single command (non-interactive)")
    parser.add_argument("--context", default="", help="context for --command")
    args = parser.parse_args()

    engine = GunaDecisionEngine()

    if args.command:
        render(engine.decide(args.command, args.context))
        return

    print("Guna action-gating demo. Ctrl-C to quit.\n")
    try:
        while True:
            command = input("command> ").strip()
            if not command:
                continue
            context = input("context> ").strip()
            render(engine.decide(command, context))
    except (KeyboardInterrupt, EOFError):
        print("\nbye.")


if __name__ == "__main__":
    main()

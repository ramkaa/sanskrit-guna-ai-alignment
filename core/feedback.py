"""
Continual-learning feedback module for the guna action-gating system.

Provides two main classes:

- ``FeedbackStore``: append-only persistence of human corrections to a JSONL
  file, CSV export compatible with the gold set, and rolling drift metrics.
- ``FeedbackLoop``: wraps a ``GunaDecisionEngine``, queues low-confidence
  decisions for human review, and can merge approved corrections back into the
  gold-labeled scenario CSV.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from loguru import logger

from core.decision import ActionDecision, GunaDecisionEngine
from core.llm_guna import Decision, Guna

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FEEDBACK_DIR = _DATA_DIR / "feedback"
_CORRECTIONS_PATH = _FEEDBACK_DIR / "corrections.jsonl"
_GOLD_SET_PATH = _DATA_DIR / "scenarios" / "robotics_scenarios.csv"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class Correction:
    """A single human correction of a predicted decision."""

    id: str
    timestamp: str
    command: str
    context: str
    predicted_guna: Guna
    predicted_decision: Decision
    predicted_confidence: float
    human_guna: Guna
    human_decision: Decision
    human_rationale: str
    reviewer_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Correction":
        return cls(**d)


@dataclass
class PendingDecision:
    """A low-confidence decision awaiting human review."""

    id: str
    timestamp: str
    command: str
    context: str
    guna: Guna
    decision: Decision
    confidence: float
    rationale: str
    model: str
    safe_default_applied: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_action_decision(cls, ad: ActionDecision) -> "PendingDecision":
        return cls(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            command=ad.command,
            context=ad.context,
            guna=ad.guna,
            decision=ad.decision,
            confidence=ad.confidence,
            rationale=ad.rationale,
            model=ad.model,
            safe_default_applied=ad.safe_default_applied,
        )


# ---------------------------------------------------------------------------
# FeedbackStore -- persistence + drift metrics
# ---------------------------------------------------------------------------
class FeedbackStore:
    """Append-only store for human corrections, backed by a JSONL file.

    Thread-safe: all mutations acquire ``_lock``.
    """

    def __init__(
        self,
        path: Path = _CORRECTIONS_PATH,
        drift_window: int = 100,
    ):
        self._path = path
        self._drift_window = drift_window
        self._lock = threading.Lock()

        # Rolling window: 1 = corrected, 0 = not corrected (accepted as-is).
        self._correction_flags: deque[int] = deque(maxlen=drift_window)

        # Total counts
        self._total_corrections: int = 0
        self._total_decisions: int = 0

        # Ensure parent dir exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing corrections to seed counters
        self._load_existing()

    # -- bootstrap --------------------------------------------------------

    def _load_existing(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._total_corrections += 1
                    self._total_decisions += 1
                    self._correction_flags.append(1)

    # -- public API -------------------------------------------------------

    def record_correction(self, correction: Correction) -> None:
        """Persist a correction and update drift metrics."""
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(correction.to_json_line() + "\n")
            self._total_corrections += 1
            self._total_decisions += 1
            self._correction_flags.append(1)
        logger.info(f"Recorded correction {correction.id} by {correction.reviewer_id}")

    def record_acceptance(self) -> None:
        """Record that a decision was accepted without correction (for drift tracking)."""
        with self._lock:
            self._total_decisions += 1
            self._correction_flags.append(0)

    def get_all_corrections(self) -> List[Correction]:
        """Return every stored correction."""
        corrections: List[Correction] = []
        if not self._path.exists():
            return corrections
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    corrections.append(Correction.from_dict(json.loads(line)))
        return corrections

    def drift_stats(self) -> Dict[str, Any]:
        """Return rolling-window and lifetime drift metrics."""
        with self._lock:
            window_size = len(self._correction_flags)
            window_corrections = sum(self._correction_flags) if window_size else 0
            correction_rate = (
                window_corrections / window_size if window_size > 0 else 0.0
            )
            return {
                "window_size": window_size,
                "window_max": self._drift_window,
                "window_corrections": window_corrections,
                "correction_rate": round(correction_rate, 4),
                "total_corrections": self._total_corrections,
                "total_decisions": self._total_decisions,
                "lifetime_correction_rate": round(
                    self._total_corrections / self._total_decisions, 4
                )
                if self._total_decisions > 0
                else 0.0,
            }

    def export_csv(self, dest: Path) -> int:
        """Export corrections in gold-set-compatible CSV format.

        Returns the number of rows written.
        """
        corrections = self.get_all_corrections()
        if not corrections:
            return 0

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "command", "context", "guna", "decision", "rationale"])
            for i, c in enumerate(corrections, start=1):
                writer.writerow([
                    i,
                    c.command,
                    c.context,
                    c.human_guna,
                    c.human_decision,
                    c.human_rationale,
                ])
        return len(corrections)


# ---------------------------------------------------------------------------
# FeedbackLoop -- wraps GunaDecisionEngine with review queue + merge
# ---------------------------------------------------------------------------
class FeedbackLoop:
    """Wraps a ``GunaDecisionEngine`` to capture low-confidence decisions for
    human review and feed corrections back into the gold set.
    """

    def __init__(
        self,
        engine: GunaDecisionEngine,
        store: Optional[FeedbackStore] = None,
        review_threshold: float = 0.7,
    ):
        self.engine = engine
        self.store = store or FeedbackStore()
        self.review_threshold = review_threshold

        # In-memory pending review queue, keyed by decision id.
        self._pending: Dict[str, PendingDecision] = {}
        self._lock = threading.Lock()

    # -- decision wrapper -------------------------------------------------

    def decide(self, command: str, context: str) -> ActionDecision:
        """Run the engine and, if confidence is low, queue for review."""
        result = self.engine.decide(command, context)
        if result.confidence < self.review_threshold:
            pending = PendingDecision.from_action_decision(result)
            with self._lock:
                self._pending[pending.id] = pending
            logger.info(
                f"Queued decision {pending.id} for review "
                f"(confidence={result.confidence:.2f} < {self.review_threshold})"
            )
        else:
            # Good-confidence decision: record as accepted for drift tracking
            self.store.record_acceptance()
        return result

    # -- review queue -----------------------------------------------------

    def pending_reviews(self) -> List[PendingDecision]:
        with self._lock:
            return list(self._pending.values())

    def get_pending(self, decision_id: str) -> Optional[PendingDecision]:
        with self._lock:
            return self._pending.get(decision_id)

    # -- correction -------------------------------------------------------

    def correct(
        self,
        decision_id: str,
        human_guna: Guna,
        human_decision: Decision,
        human_rationale: str,
        reviewer_id: str,
    ) -> Correction:
        """Submit a human correction for a pending (or any) decision.

        If the ``decision_id`` matches a pending review it is removed from the
        queue; otherwise the correction is still recorded (humans may correct
        any past decision).
        """
        with self._lock:
            pending = self._pending.pop(decision_id, None)

        if pending is None:
            raise KeyError(
                f"No pending decision with id '{decision_id}'. "
                "It may have already been reviewed or the id is incorrect."
            )

        correction = Correction(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            command=pending.command,
            context=pending.context,
            predicted_guna=pending.guna,
            predicted_decision=pending.decision,
            predicted_confidence=pending.confidence,
            human_guna=human_guna,
            human_decision=human_decision,
            human_rationale=human_rationale,
            reviewer_id=reviewer_id,
        )
        self.store.record_correction(correction)
        return correction

    # -- merge into gold set ----------------------------------------------

    def merge_corrections_into_gold(
        self,
        gold_path: Path = _GOLD_SET_PATH,
    ) -> int:
        """Append approved corrections to the gold-labeled scenario CSV.

        Returns the number of rows appended.
        """
        corrections = self.store.get_all_corrections()
        if not corrections:
            return 0

        # Determine the next id by reading the current gold set
        next_id = 1
        if gold_path.exists():
            with open(gold_path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        next_id = max(next_id, int(row["id"]) + 1)
                    except (ValueError, KeyError):
                        pass

        # Build a set of (command, context) pairs already in the gold set to
        # avoid duplicates.
        existing_pairs: set[tuple[str, str]] = set()
        if gold_path.exists():
            with open(gold_path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    existing_pairs.add((row.get("command", ""), row.get("context", "")))

        appended = 0
        with open(gold_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for c in corrections:
                pair = (c.command, c.context)
                if pair in existing_pairs:
                    continue
                writer.writerow([
                    next_id,
                    c.command,
                    c.context,
                    c.human_guna,
                    c.human_decision,
                    c.human_rationale,
                ])
                existing_pairs.add(pair)
                next_id += 1
                appended += 1

        logger.info(f"Merged {appended} corrections into {gold_path}")
        return appended

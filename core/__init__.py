"""Core package.

Submodules are imported lazily so that importing the lightweight decision layer
(`core.decision`) does not pull in the sklearn/numpy baseline classifier, and
vice versa.
"""

__all__ = ["GunaClassifier", "GunaDecisionEngine", "GunaReasoner"]


def __getattr__(name):
    if name == "GunaClassifier":
        from .classifier import GunaClassifier

        return GunaClassifier
    if name == "GunaDecisionEngine":
        from .decision import GunaDecisionEngine

        return GunaDecisionEngine
    if name == "GunaReasoner":
        from .llm_guna import GunaReasoner

        return GunaReasoner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

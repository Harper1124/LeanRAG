"""LeanRAG Phase 3 contracts and the Step 0-2 build pipeline."""

from typing import Any

from .schema import CanonicalEntity, CanonicalRelation, MediaSemanticUnit, SourceReference

__all__ = [
    "CanonicalEntity",
    "CanonicalRelation",
    "MediaSemanticUnit",
    "SourceReference",
    "run_phase3",
]

__version__ = "0.1.0"


def run_phase3(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazily load the pipeline so ``python -m ...pipeline`` stays warning-free."""
    from .pipeline import run_phase3 as _run_phase3

    return _run_phase3(*args, **kwargs)

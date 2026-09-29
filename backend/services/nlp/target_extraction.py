"""Deterministic extraction of natural-language target references."""

import re
from typing import Any


_TARGET_VERBS = "predict|estimate|forecast|classify|identify|determine"
_DATASET_CONTEXT = re.compile(r"\s+(?:using|with|from)\s+(?:this|the)\s+dataset\b.*$", re.IGNORECASE)
_QUESTION_START = re.compile(
    r"^(?:will|would|can|could|should|may|might|must|do|does|did|is|are|was|were|has|have|had)\b.+",
    re.IGNORECASE,
)
_WH_QUESTION_START = re.compile(
    r"^(?:how\s+(?:much|many)|what\s+(?:will|would|is|are|was|were))\b.+",
    re.IGNORECASE,
)
_ACTION_ONLY_ENDING = re.compile(r"\b(?:predict|estimate|forecast|classify|identify|determine)$", re.IGNORECASE)


def extract_target_reference(text: str) -> dict[str, Any]:
    """Extract a natural-language target reference without inspecting data."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must be a non-empty string")

    request = text.strip().strip(".!?").strip()
    match = re.match(rf"^(?:{_TARGET_VERBS})\s+(.+)$", request, re.IGNORECASE)
    if match is not None:
        # Keep the phrase after the action verb and remove dataset context.
        target = _DATASET_CONTEXT.sub("", match.group(1)).strip()
        target = re.sub(r"^the\s+", "", target, flags=re.IGNORECASE).strip()
    elif _QUESTION_START.match(request) or _WH_QUESTION_START.match(request):
        target = request
    elif re.search(r"\blikely\s+to\b", request, re.IGNORECASE):
        target = request
    else:
        return _no_target_result()

    if (
        not target
        or target.lower() in {"something", "something useful", "this dataset"}
        or _ACTION_ONLY_ENDING.search(target)
    ):
        return _no_target_result()

    return {
        "target_reference": target.lower(),
        "target_found": True,
        "needs_clarification": False,
    }


def _no_target_result() -> dict[str, Any]:
    """Return the standard result when no target can be extracted."""
    return {
        "target_reference": None,
        "target_found": False,
        "needs_clarification": True,
    }

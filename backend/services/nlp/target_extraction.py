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
_COLUMN_TOKEN = r"[`'\"]?([\w][\w.-]*)[`'\"]?"
_TARGET_NAME_QUALIFIER = r"(?:(?:strictly|specifically|explicitly)\s+)?"
_EXPLICIT_TARGET_PATTERNS = (
    re.compile(
        rf"\busing\s+(?:the\s+)?{_COLUMN_TOKEN}\s+as\s+(?:the\s+)?target(?:\s+column)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bwith\s+(?:the\s+)?{_COLUMN_TOKEN}\s+as\s+(?:the\s+)?target(?:\s+column)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:use|select|choose|set)\s+(?:the\s+)?{_COLUMN_TOKEN}\s+as\s+(?:the\s+)?target(?:\s+column)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:target|label)(?:\s+(?:column|field))?\s*(?:is|:|=)\s*{_TARGET_NAME_QUALIFIER}{_COLUMN_TOKEN}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:in|from)\s+(?:the\s+)?{_COLUMN_TOKEN}\s+column\b",
        re.IGNORECASE,
    ),
)
_USE_TARGET_TO_PREDICT = re.compile(
    r"^use\s+(?:the\s+)?(.+?)\s+to\s+predict$",
    re.IGNORECASE,
)


def extract_target_reference(text: str) -> dict[str, Any]:
    """Extract a natural-language target reference without inspecting data."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must be a non-empty string")

    request = text.strip().strip(".!?").strip()
    for pattern in _EXPLICIT_TARGET_PATTERNS:
        match = pattern.search(request)
        if match is not None:
            return {
                "target_reference": match.group(1).lower(),
                "target_found": True,
                "needs_clarification": False,
                "explicit_target_column": True,
            }

    use_target_match = _USE_TARGET_TO_PREDICT.match(request)
    if use_target_match is not None:
        target = use_target_match.group(1).strip()
    else:
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

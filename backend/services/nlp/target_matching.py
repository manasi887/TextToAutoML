"""Deterministic matching of natural-language targets to DataFrame columns."""

import re
from typing import Any

import pandas as pd


MIN_MATCH_SCORE = 0.80
"""Minimum candidate score required for automatic column selection."""

MIN_SCORE_MARGIN = 0.25
"""Minimum gap required between the best and second-best candidates."""

MIN_CANDIDATE_SCORE = 0.50
"""Minimum score for a column to be included as a meaningful candidate."""

_SEPARATOR_PATTERN = re.compile(r"[_\-\s]+")


def _normalize(value: Any) -> str:
    """Normalize a target or column name without changing the source value."""
    return _SEPARATOR_PATTERN.sub(" ", str(value).strip().lower()).strip()


def _score_match(target: str, column: str) -> float:
    """Score exact, token-overlap, and token-containment matches."""
    if not target or not column:
        return 0.0
    if target == column:
        return 1.0

    target_tokens = set(target.split())
    column_tokens = set(column.split())
    overlap = target_tokens.intersection(column_tokens)
    if not overlap:
        return 0.0

    # Coverage rewards all target words while keeping non-exact matches below 1.
    token_coverage = len(overlap) / len(target_tokens)
    return 0.8 * token_coverage


def match_target_column(target_reference: str, df: pd.DataFrame) -> dict[str, Any]:
    """Match a natural-language target reference to a DataFrame column."""
    if not isinstance(target_reference, str):
        raise TypeError("target_reference must be a string")
    if not target_reference.strip():
        raise ValueError("target_reference must be a non-empty string")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if len(df.columns) == 0:
        raise ValueError("df must contain at least one column")

    normalized_target = _normalize(target_reference)
    scored_candidates: list[dict[str, Any]] = []
    for original_column in df.columns:
        normalized_column = _normalize(original_column)
        score = _score_match(normalized_target, normalized_column)
        if score >= MIN_CANDIDATE_SCORE:
            scored_candidates.append(
                {"column": str(original_column), "score": float(score)}
            )

    candidates = sorted(
        scored_candidates,
        key=lambda candidate: (-candidate["score"], candidate["column"]),
    )
    if not candidates:
        return {
            "target_reference": target_reference,
            "matched_column": None,
            "confidence": 0.0,
            "candidates": [],
            "needs_clarification": True,
        }

    best = candidates[0]
    second_score = candidates[1]["score"] if len(candidates) > 1 else None
    has_required_margin = (
        second_score is None or best["score"] - second_score >= MIN_SCORE_MARGIN
    )
    is_clear_match = best["score"] >= MIN_MATCH_SCORE and has_required_margin

    return {
        "target_reference": target_reference,
        "matched_column": best["column"] if is_clear_match else None,
        "confidence": best["score"] if is_clear_match else 0.0,
        "candidates": candidates,
        "needs_clarification": not is_clear_match,
    }

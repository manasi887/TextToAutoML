"""Map confirmed NLP intents to AutoML problem types."""

import re
from typing import Any


_CLASSIFICATION_PATTERNS = (
	re.compile(r"\bbinary\s+classification\b", re.IGNORECASE),
	re.compile(r"\bclassification\s+(?:task|problem|analysis)\b", re.IGNORECASE),
	re.compile(r"\b(?:classify|categorize|categorise)\b", re.IGNORECASE),
)
_REGRESSION_PATTERNS = (
	re.compile(r"\bregression\s+(?:task|problem|analysis)\b", re.IGNORECASE),
	re.compile(r"\b(?:use|apply|run|perform|do)\s+(?:a\s+)?regression\b", re.IGNORECASE),
	re.compile(r"\b(?:numerical|numeric|continuous)\s+(?:value|target|outcome)\b", re.IGNORECASE),
)
_NEGATED_INTENT_PREFIX = re.compile(
	r"\b(?:not|never|without|don't|do not|rather than|instead of)\s+(?:(?:a|an|the)\s+)?$",
	re.IGNORECASE,
)


def _has_unnegated_match(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
	for pattern in patterns:
		for match in pattern.finditer(text):
			prefix = text[max(0, match.start() - 24):match.start()]
			if not _NEGATED_INTENT_PREFIX.search(prefix):
				return True
	return False


def detect_explicit_task_intent(text: str) -> str | None:
	"""Detect only directly stated classification or regression requests."""
	requests_classification = _has_unnegated_match(_CLASSIFICATION_PATTERNS, text)
	requests_regression = _has_unnegated_match(_REGRESSION_PATTERNS, text)
	if requests_classification and requests_regression:
		return "conflict"
	if requests_classification:
		return "classification"
	if requests_regression:
		return "regression"
	return None


_INTENT_TO_PROBLEM_TYPE: dict[str, str | None] = {
	"classification": "classification",
	"regression": "regression",
	"clustering": "clustering",
	"unknown": None,
}


def map_intent_to_task(intent: str) -> dict[str, Any]:
	"""Convert a supported NLP intent into the AutoML task terminology."""
	if intent not in _INTENT_TO_PROBLEM_TYPE:
		raise ValueError(
			f"Unsupported intent '{intent}'. Expected one of: "
			f"{', '.join(_INTENT_TO_PROBLEM_TYPE)}"
		)

	problem_type = _INTENT_TO_PROBLEM_TYPE[intent]
	return {
		"intent": intent,
		"problem_type": problem_type,
		"clarification_required": intent == "unknown",
	}

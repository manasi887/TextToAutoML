"""Map confirmed NLP intents to AutoML problem types."""

from typing import Any


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

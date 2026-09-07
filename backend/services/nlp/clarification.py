"""Confidence and clarification decisions for intent predictions."""

import math
from collections.abc import Mapping
from typing import Any


MIN_TOP_PROBABILITY = 0.30
"""Minimum top-intent probability for this conservative baseline."""

MIN_TOP_TWO_MARGIN = 0.05
"""Minimum margin between the top two intents for a decisive prediction."""

CLARIFICATION_MESSAGE = (
    "Could you clarify what you would like to do with the dataset? For example, "
    "predict a category, predict a numerical value, or group similar records?"
)


def _validate_intent_result(intent_result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the required shape and probability values of an intent result."""
    if not isinstance(intent_result, Mapping):
        raise TypeError("intent_result must be a mapping")

    required_fields = {"intent", "confidence", "probabilities"}
    missing_fields = required_fields.difference(intent_result)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"intent_result is missing required field(s): {missing}")

    if not isinstance(intent_result["intent"], str) or not intent_result["intent"]:
        raise ValueError("intent_result['intent'] must be a non-empty string")

    probabilities = intent_result["probabilities"]
    if not isinstance(probabilities, Mapping) or len(probabilities) < 2:
        raise ValueError("intent_result['probabilities'] must contain at least two intents")

    for intent, probability in probabilities.items():
        if not isinstance(intent, str) or not intent:
            raise ValueError("probability intent names must be non-empty strings")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f"probability for '{intent}' must be numeric")
        if not math.isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
            raise ValueError(f"probability for '{intent}' must be between 0 and 1")

    confidence = intent_result["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("intent_result['confidence'] must be numeric")
    if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("intent_result['confidence'] must be between 0 and 1")

    return intent_result


def analyze_intent_confidence(intent_result: Mapping[str, Any]) -> dict[str, Any]:
    """Decide whether an intent prediction can proceed or needs clarification.

    The thresholds are conservative starting points, not scientifically calibrated
    values. Both top probability and top-two margin must meet their thresholds.
    """
    validated_result = _validate_intent_result(intent_result)
    probabilities = validated_result["probabilities"]
    ranked_intents = sorted(
        probabilities.items(), key=lambda item: float(item[1]), reverse=True
    )
    top_intent, top_probability = ranked_intents[0]
    second_intent, second_probability = ranked_intents[1]
    top_probability = float(top_probability)
    second_probability = float(second_probability)
    margin = top_probability - second_probability

    reasons: list[str] = []
    if top_intent == "unknown":
        reasons.append("the predicted intent is unknown")
    if top_probability < MIN_TOP_PROBABILITY:
        reasons.append(
            f"top probability {top_probability:.4f} is below "
            f"{MIN_TOP_PROBABILITY:.2f}"
        )
    if margin < MIN_TOP_TWO_MARGIN:
        reasons.append(
            f"top-two margin {margin:.4f} is below {MIN_TOP_TWO_MARGIN:.2f}"
        )

    needs_clarification = bool(reasons)
    clarification_reason = (
        f"{' and '.join(reasons).capitalize()}. {CLARIFICATION_MESSAGE}"
        if needs_clarification
        else "The prediction meets the baseline confidence and margin requirements."
    )

    return {
        "intent": validated_result["intent"],
        "confidence": float(validated_result["confidence"]),
        "top_intent": top_intent,
        "top_probability": top_probability,
        "second_intent": second_intent,
        "second_probability": second_probability,
        "margin": margin,
        "needs_clarification": needs_clarification,
        "clarification_reason": clarification_reason,
    }

"""Inference utilities for the trained user-intent classifier."""

import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path(
	os.getenv(
		"INTENT_MODEL_DIR",
		Path(__file__).resolve().parents[3]
		/ "backend"
		/ "storage"
		/ "nlp_models"
		/ "intent_classifier",
	)
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer: Any = None
_model: Any = None
_label_mapping: dict[str, str] | None = None


def _load_artifacts() -> tuple[Any, Any, dict[str, str]]:
	"""Load and cache the tokenizer, model, and intent labels."""
	global _tokenizer, _model, _label_mapping

	if _tokenizer is not None and _model is not None and _label_mapping is not None:
		return _tokenizer, _model, _label_mapping

	if not MODEL_DIR.is_dir():
		raise FileNotFoundError(
			f"Intent classifier model directory was not found: {MODEL_DIR}"
		)

	mapping_path = MODEL_DIR / "label_mapping.json"
	if not mapping_path.is_file():
		raise FileNotFoundError(f"Intent classifier label mapping was not found: {mapping_path}")

	try:
		_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
		_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
		mapping_data = json.loads(mapping_path.read_text(encoding="utf-8"))
		_label_mapping = {
			str(class_id): str(intent)
			for class_id, intent in mapping_data["id_to_label"].items()
		}
	except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as error:
		raise RuntimeError(
			f"Could not load the intent classifier from {MODEL_DIR}: {error}"
		) from error

	_model.to(DEVICE)
	_model.eval()
	return _tokenizer, _model, _label_mapping


def detect_user_intent(text: str) -> dict[str, Any]:
	"""Predict the intent of a non-empty user message."""
	if not isinstance(text, str) or not text.strip():
		raise ValueError("text must be a non-empty string")

	tokenizer, model, label_mapping = _load_artifacts()
	inputs = tokenizer(text, return_tensors="pt", truncation=True)
	inputs = {name: value.to(DEVICE) for name, value in inputs.items()}

	with torch.no_grad():
		logits = model(**inputs).logits
		probabilities = torch.softmax(logits, dim=-1)

	class_id = int(torch.argmax(probabilities, dim=-1).item())
	intent = label_mapping.get(str(class_id))
	if intent is None:
		raise RuntimeError(f"No intent mapping found for predicted class ID {class_id}")
	probability_distribution = {
		label_mapping[str(index)]: float(probabilities[0, index].item())
		for index in range(probabilities.shape[-1])
	}

	return {
		"intent": intent,
		"confidence": probability_distribution[intent],
		"probabilities": probability_distribution,
	}

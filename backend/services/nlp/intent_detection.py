"""Inference utilities for the trained user-intent classifier."""

import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import INTENT_MODEL_DIR


MODEL_DIR = INTENT_MODEL_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer: Any = None
_model: Any = None
_label_mapping: dict[str, str] | None = None
_temperature: float | None = None


class IntentModelUnavailableError(FileNotFoundError):
	"""Raised when the configured intent classifier cannot be loaded."""


def inspect_model_artifacts(model_dir: Path | None = None) -> dict[str, Any]:
	"""Check required intent artifacts without initializing the model weights."""
	artifact_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
	missing: list[str] = []
	config: dict[str, Any] | None = None
	if not artifact_dir.is_dir():
		missing.append("model directory")
	else:
		for filename in ("config.json", "tokenizer_config.json"):
			path = artifact_dir / filename
			if not path.is_file() or path.stat().st_size == 0:
				missing.append(filename)
				continue
			try:
				payload = json.loads(path.read_text(encoding="utf-8"))
			except (OSError, UnicodeDecodeError, json.JSONDecodeError):
				missing.append(f"valid {filename}")
				continue
			if not isinstance(payload, dict):
				missing.append(f"valid {filename}")
			elif filename == "config.json":
				config = payload

		if not any(
			(artifact_dir / filename).is_file()
			and (artifact_dir / filename).stat().st_size > 0
			for filename in ("model.safetensors", "pytorch_model.bin")
		):
			missing.append("model weights")
		if not any(
			(artifact_dir / filename).is_file()
			and (artifact_dir / filename).stat().st_size > 0
			for filename in ("tokenizer.json", "vocab.txt")
		):
			missing.append("tokenizer vocabulary")

		mapping_path = artifact_dir / "label_mapping.json"
		if mapping_path.is_file() and mapping_path.stat().st_size > 0:
			try:
				mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
			except (OSError, UnicodeDecodeError, json.JSONDecodeError):
				missing.append("valid label mapping")
			else:
				labels = mapping.get("id_to_label") if isinstance(mapping, dict) else None
				if (
					not isinstance(labels, dict)
					or not labels
					or any(not isinstance(key, str) or not isinstance(value, str) for key, value in labels.items())
				):
					missing.append("valid label mapping")
				elif config is not None and config.get("num_labels") not in (None, len(labels)):
					missing.append("matching model and label counts")

	return {"available": not missing, "missing": missing}


def _load_temperature() -> float:
	"""Return the persisted calibration temperature or the neutral default."""
	global _temperature
	if _temperature is not None:
		return _temperature

	temperature_path = MODEL_DIR / "temperature.json"
	if not temperature_path.is_file():
		_temperature = 1.0
		return _temperature

	try:
		payload = json.loads(temperature_path.read_text(encoding="utf-8"))
		if not isinstance(payload, dict):
			raise ValueError("temperature artifact must contain a JSON object")
		value = payload.get("temperature", payload.get("value", 1.0))
		if isinstance(value, bool):
			raise ValueError("temperature must be numeric")
		temperature = float(value)
	except (OSError, json.JSONDecodeError, TypeError, ValueError):
		temperature = 1.0

	if not math.isfinite(temperature) or temperature <= 0:
		temperature = 1.0

	_temperature = temperature
	return _temperature


def _load_artifacts() -> tuple[Any, Any, dict[str, str]]:
	"""Load and cache the tokenizer, model, and intent labels."""
	global _tokenizer, _model, _label_mapping

	if _tokenizer is not None and _model is not None and _label_mapping is not None:
		return _tokenizer, _model, _label_mapping

	artifact_status = inspect_model_artifacts()
	if not artifact_status["available"]:
		raise IntentModelUnavailableError(
			"Intent classifier artifacts are missing or incomplete: "
			+ ", ".join(artifact_status["missing"])
		)

	mapping_path = MODEL_DIR / "label_mapping.json"

	try:
		_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
		_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
		_model.to(DEVICE)
		_model.eval()
		mapping_data = json.loads(mapping_path.read_text(encoding="utf-8"))
		_label_mapping = {
			str(class_id): str(intent)
			for class_id, intent in mapping_data["id_to_label"].items()
		}
	except (ImportError, KeyError, OSError, RuntimeError, json.JSONDecodeError, TypeError, ValueError) as error:
		raise IntentModelUnavailableError(
			"Intent classifier artifacts are present but could not be initialized."
		) from error

	return _tokenizer, _model, _label_mapping


def detect_user_intent(text: str) -> dict[str, Any]:
	"""Predict the top intent class and confidence for a user request."""
	if not isinstance(text, str) or not text.strip():
		raise ValueError("text must be a non-empty string")

	tokenizer, model, label_mapping = _load_artifacts()
	inputs = tokenizer(text, return_tensors="pt", truncation=True)
	inputs = {name: value.to(DEVICE) for name, value in inputs.items()}

	with torch.no_grad():
		logits = model(**inputs).logits
		temperature = _load_temperature()
		probabilities = torch.softmax(logits / temperature, dim=-1)

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

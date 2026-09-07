"""Fine-tune DistilBERT for TextToAutoML intent classification."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

DATASET_PATH = Path(__file__).resolve().parent / "data" / "intent_dataset.csv"
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "storage" / "nlp_models" / "intent_classifier"
)
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 4
SEED = 42
LABEL_NAMES = ("classification", "regression", "clustering", "unknown")
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {str(index): label for index, label in enumerate(LABEL_NAMES)}


def _load_dependencies() -> dict[str, Any]:
    """Load runtime dependencies with an actionable error when they are absent."""
    missing: list[str] = []
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
        np = None
    try:
        import pandas as pd
    except ImportError:
        missing.append("pandas")
        pd = None
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        missing.append("torch")
        torch = None
        Dataset = None
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            precision_recall_fscore_support,
        )
        from sklearn.model_selection import train_test_split
    except ImportError:
        missing.append("scikit-learn")
        accuracy_score = classification_report = precision_recall_fscore_support = None
        train_test_split = None
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError:
        missing.append("transformers")
        AutoModelForSequenceClassification = AutoTokenizer = Trainer = None
        TrainingArguments = set_seed = EarlyStoppingCallback = None

    if missing:
        packages = ", ".join(dict.fromkeys(missing))
        raise ImportError(
            f"Missing required dependencies: {packages}. "
            "Install them in the active environment before running this script."
        )

    return {
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "Dataset": Dataset,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "np": np,
        "pd": pd,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "set_seed": set_seed,
        "torch": torch,
        "train_test_split": train_test_split,
    }


def load_intent_dataset(path: Path = DATASET_PATH, *, pd: Any = None) -> Any:
    """Load and validate the intent CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Intent dataset not found: {path}")
    if pd is None:
        raise RuntimeError("pandas is required to load the intent dataset")

    dataframe = pd.read_csv(path)
    if list(dataframe.columns) != ["text", "intent"]:
        raise ValueError(
            "Dataset must contain exactly these columns in this order: text, intent"
        )
    if dataframe.empty:
        raise ValueError("Intent dataset is empty")
    if dataframe["text"].isna().any() or dataframe["intent"].isna().any():
        raise ValueError("Dataset cannot contain missing text or intent values")
    if not dataframe["text"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Every text value must be a non-empty string")

    observed_labels = set(dataframe["intent"].unique())
    invalid_labels = observed_labels.difference(LABEL_TO_ID)
    if invalid_labels:
        raise ValueError(
            f"Invalid intent labels: {sorted(invalid_labels)}. "
            f"Allowed labels: {list(LABEL_NAMES)}"
        )
    dataframe = dataframe.copy()
    dataframe["label"] = dataframe["intent"].map(LABEL_TO_ID).astype("int64")
    return dataframe


def split_dataset(dataframe: Any, *, train_test_split: Any, seed: int = SEED) -> tuple[Any, Any, Any]:
    """Create 70/15/15 stratified train, validation, and test splits."""
    counts = dataframe["label"].value_counts()
    if len(counts) != NUM_LABELS or counts.min() < 2:
        raise ValueError(
            "Dataset does not contain enough examples per class for safe "
            "70/15/15 stratified splitting across both split stages."
        )

    try:
        train, temporary = train_test_split(
            dataframe,
            test_size=0.30,
            stratify=dataframe["label"],
            random_state=seed,
        )
        validation, test = train_test_split(
            temporary,
            test_size=0.50,
            stratify=temporary["label"],
            random_state=seed,
        )
    except ValueError as error:
        raise ValueError(
            "Dataset does not contain enough examples per class for safe "
            "70/15/15 stratified splitting across both split stages."
        ) from error
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def tokenize_dataset(dataframe: Any, tokenizer: Any, dataset_type: Any) -> Any:
    """Tokenize a dataframe and expose it as a PyTorch dataset."""
    encodings = tokenizer(
        dataframe["text"].tolist(),
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    labels = dataframe["label"].tolist()

    class IntentDataset(dataset_type):
        def __len__(self) -> int:
            return len(labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {key: values[index] for key, values in encodings.items()}
            item["labels"] = labels[index]
            return item

    return IntentDataset()


def compute_metrics(eval_prediction: Any, metrics: dict[str, Any]) -> dict[str, float]:
    """Calculate classification metrics for Hugging Face Trainer evaluation."""
    predictions = metrics["np"].argmax(eval_prediction.predictions, axis=-1)
    actual = eval_prediction.label_ids
    precision, recall, f1, _ = metrics["precision_recall_fscore_support"](
        actual,
        predictions,
        labels=list(range(NUM_LABELS)),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(metrics["accuracy_score"](actual, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def build_model(auto_model: Any, model_name: str = MODEL_NAME) -> Any:
    """Build a four-class sequence classifier from the pretrained checkpoint."""
    return auto_model.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label={index: label for index, label in enumerate(LABEL_NAMES)},
        label2id=LABEL_TO_ID,
    )


def train_model(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    validation_dataset: Any,
    trainer_type: Any,
    training_arguments_type: Any,
    early_stopping_callback_type: Any,
    compute_metrics_fn: Any,
    output_dir: Path = OUTPUT_DIR,
) -> Any:
    """Fine-tune the model with validation evaluation each epoch."""
    training_arguments = training_arguments_type(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=10,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="epoch",
        seed=SEED,
        report_to="none",
    )
    trainer = trainer_type(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=[early_stopping_callback_type(early_stopping_patience=2)],
    )
    trainer.train()
    return trainer


def evaluate_model(trainer: Any, test_dataset: Any, test_dataframe: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the held-out test split and return aggregate and per-class metrics."""
    prediction_output = trainer.predict(test_dataset)
    predictions = metrics["np"].argmax(prediction_output.predictions, axis=-1)
    actual = test_dataframe["label"].to_numpy()
    precision, recall, f1, _ = metrics["precision_recall_fscore_support"](
        actual,
        predictions,
        labels=list(range(NUM_LABELS)),
        average="macro",
        zero_division=0,
    )
    report = metrics["classification_report"](
        actual,
        predictions,
        labels=list(range(NUM_LABELS)),
        target_names=list(LABEL_NAMES),
        zero_division=0,
    )
    return {
        "accuracy": float(metrics["accuracy_score"](actual, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "classification_report": report,
    }


def save_model(trainer: Any, tokenizer: Any, output_dir: Path = OUTPUT_DIR) -> None:
    """Save model, tokenizer, and inference label mapping."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    mapping = {
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
    }
    (output_dir / "label_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n",
        encoding="utf-8",
    )


def _print_distribution(name: str, dataframe: Any) -> None:
    distribution = {label: int((dataframe["intent"] == label).sum()) for label in LABEL_NAMES}
    print(f"{name} class distribution: {distribution}")


def main() -> None:
    print(f"Loading intent dataset: {DATASET_PATH}")
    dependencies = _load_dependencies()
    dependencies["set_seed"](SEED)
    random.seed(SEED)
    dependencies["np"].random.seed(SEED)
    dependencies["torch"].manual_seed(SEED)

    dataframe = load_intent_dataset(pd=dependencies["pd"])
    train, validation, test = split_dataset(
        dataframe,
        train_test_split=dependencies["train_test_split"],
    )
    print(f"Dataset size: {len(dataframe)}")
    print(f"Train/validation/test sizes: {len(train)}/{len(validation)}/{len(test)}")
    _print_distribution("Train", train)
    _print_distribution("Validation", validation)
    _print_distribution("Test", test)
    print(f"Model: {MODEL_NAME}")
    print(f"Number of labels: {NUM_LABELS}")
    print("Training configuration: epochs=10, early_stopping_patience=2, learning_rate=2e-5, train_batch_size=8, eval_batch_size=8, weight_decay=0.01")

    tokenizer = dependencies["AutoTokenizer"].from_pretrained(MODEL_NAME)
    train_dataset = tokenize_dataset(train, tokenizer, dependencies["Dataset"])
    validation_dataset = tokenize_dataset(validation, tokenizer, dependencies["Dataset"])
    test_dataset = tokenize_dataset(test, tokenizer, dependencies["Dataset"])
    model = build_model(dependencies["AutoModelForSequenceClassification"])
    trainer = train_model(
        model,
        tokenizer,
        train_dataset,
        validation_dataset,
        dependencies["Trainer"],
        dependencies["TrainingArguments"],
        dependencies["EarlyStoppingCallback"],
        lambda eval_prediction: compute_metrics(
            eval_prediction,
            {
                "accuracy_score": dependencies["accuracy_score"],
                "np": dependencies["np"],
                "precision_recall_fscore_support": dependencies[
                    "precision_recall_fscore_support"
                ],
            },
        ),
    )
    results = evaluate_model(
        trainer,
        test_dataset,
        test,
        {
            "accuracy_score": dependencies["accuracy_score"],
            "classification_report": dependencies["classification_report"],
            "np": dependencies["np"],
            "precision_recall_fscore_support": dependencies["precision_recall_fscore_support"],
        },
    )
    save_model(trainer, tokenizer)
    print(f"Final test accuracy: {results['accuracy']:.4f}")
    print(f"Final macro precision: {results['precision_macro']:.4f}")
    print(f"Final macro recall: {results['recall_macro']:.4f}")
    print(f"Final macro F1: {results['f1_macro']:.4f}")
    print("Classification report:")
    print(results["classification_report"])
    print(f"Saved model and tokenizer to: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ImportError, OSError, ValueError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

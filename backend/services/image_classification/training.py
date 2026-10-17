from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights

from .dataset import validate_split_for_leakage
from .model import build_mobilenet_v2_classifier


class ImagePathDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str],
        class_to_index: dict[str, int],
        image_labels: dict[str, str],
        transform: transforms.Compose | None = None,
    ):
        self.image_paths = list(image_paths)
        self.class_to_index = class_to_index
        self.image_labels = image_labels
        self.transform = transform or transforms.Compose([])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = Path(self.image_paths[idx])
        label_name = self.image_labels[str(image_path.resolve())]
        label = self.class_to_index[label_name]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def _build_preprocessing(image_size: int = 224) -> dict[str, Any]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return {
        "image_size": image_size,
        "channels": 3,
        "normalization_mean": mean,
        "normalization_std": std,
        "train_transform": transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        "eval_transform": transforms.Compose([
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
    }


def _evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total else 0.0
    return total_loss / total if total else 0.0, accuracy


def train_image_classifier(
    dataset_path: str | Path,
    class_to_index: dict[str, int],
    split: dict[str, list[str]],
    image_labels: dict[str, str],
    *,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seed: int = 42,
    pretrained: bool = True,
    image_size: int = 224,
) -> dict[str, Any]:
    """Train a small MobileNetV2 classifier on the extracted dataset and return metrics."""
    if not split.get("train") or not split.get("validation") or not split.get("test"):
        raise ValueError("Training, validation, and test split lists must all be present.")
    for class_name in class_to_index:
        missing_splits = [
            split_name
            for split_name in ("train", "validation", "test")
            if not any(
                image_labels.get(image_path) == class_name
                for image_path in split[split_name]
            )
        ]
        if missing_splits:
            raise ValueError(
                f"Class '{class_name}' has no readable images in the "
                f"{', '.join(missing_splits)} split(s)."
            )
    validate_split_for_leakage(split, image_labels)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    preprocessing = _build_preprocessing(image_size=image_size)
    model = build_mobilenet_v2_classifier(num_classes=len(class_to_index), pretrained=pretrained, freeze_features=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = ImagePathDataset(
        split["train"], class_to_index, image_labels, transform=preprocessing["train_transform"]
    )
    validation_dataset = ImagePathDataset(
        split["validation"], class_to_index, image_labels, transform=preprocessing["eval_transform"]
    )
    test_dataset = ImagePathDataset(
        split["test"], class_to_index, image_labels, transform=preprocessing["eval_transform"]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    training_start = time.time()
    training_losses: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        training_losses.append(epoch_loss / len(train_dataset))

        validation_loss, validation_accuracy = _evaluate_model(model, validation_loader, device)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

    test_loss, test_accuracy = _evaluate_model(model, test_loader, device)
    training_duration = time.time() - training_start

    return {
        "model_name": "MobileNetV2",
        "architecture": "MobileNetV2",
        "num_classes": len(class_to_index),
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "training_time_seconds": float(training_duration),
        "training_loss": float(training_losses[-1]) if training_losses else 0.0,
        "validation_loss": float(validation_losses[-1]) if validation_losses else 0.0,
        "validation_accuracy": float(validation_accuracies[-1]) if validation_accuracies else 0.0,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "pretrained_weights_loaded": bool(model.pretrained_weights_loaded),
        "preprocessing": preprocessing,
        "model": model,
    }

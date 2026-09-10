"""Lightweight CNN model builder for image classification.

This module builds a compact TensorFlow/Keras CNN intended for CPU-friendly
training on normalized RGB image datasets. It is intentionally independent from
other project modules and does not perform training or inference.
"""

from __future__ import annotations

import logging
from typing import Sequence

import tensorflow as tf

logger = logging.getLogger(__name__)


def build_cnn_model(
    input_shape: tuple[int, int, int],
    num_classes: int,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.30,
) -> tf.keras.Model:
    """Build and compile a lightweight CNN for image classification.

    Args:
        input_shape: Shape of the input images as (height, width, channels).
        num_classes: Number of target classes. Must be at least 2.
        learning_rate: Learning rate for the Adam optimizer.
        dropout_rate: Dropout probability applied between layers.

    Returns:
        A compiled Keras model ready to be trained.

    Raises:
        ValueError: If any parameter violates the expected constraints.
    """
    if not isinstance(input_shape, (tuple, list)):
        raise ValueError(
            "input_shape must be a tuple of exactly three positive integers, "
            f"received: {input_shape!r}"
        )

    if len(input_shape) != 3:
        raise ValueError(
            "input_shape must contain exactly three values: "
            f"(height, width, channels); received: {input_shape!r}"
        )

    if any(not isinstance(value, int) or value <= 0 for value in input_shape):
        raise ValueError(
            "input_shape values must all be positive integers; "
            f"received: {input_shape!r}"
        )

    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(
            f"num_classes must be an integer greater than or equal to 2; received: {num_classes!r}"
        )

    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError(
            "learning_rate must be a positive number; "
            f"received: {learning_rate!r}"
        )

    if not isinstance(dropout_rate, (int, float)) or not 0 <= dropout_rate < 1:
        raise ValueError(
            "dropout_rate must be greater than or equal to 0 and less than 1; "
            f"received: {dropout_rate!r}"
        )

    logger.info(
        "Building image classification CNN with input_shape=%s, num_classes=%s, "
        "learning_rate=%s, dropout_rate=%s",
        input_shape,
        num_classes,
        learning_rate,
        dropout_rate,
    )

    inputs = tf.keras.Input(shape=input_shape, name="input_images")

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv_1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_1")(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv_2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_2")(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv_3",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_3")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_3")(x)

    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="image_classification_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    logger.info("CNN model built and compiled successfully: %s", model.name)
    return model


__all__ = ["build_cnn_model"]

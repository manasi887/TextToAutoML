from pathlib import Path
import sys

import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from services.cnn.model import build_cnn_model


def test_build_cnn_model_creates_compiled_model_and_produces_softmax_predictions():
    model = build_cnn_model((32, 32, 3), 2)

    assert isinstance(model, tf.keras.Model)
    assert model.optimizer is not None
    assert model.compiled_loss is not None
    assert model.compiled_metrics is not None

    batch = tf.zeros((2, 32, 32, 3), dtype=tf.float32)
    outputs = model(batch)

    assert outputs.shape == (2, 2)
    row_sums = tf.reduce_sum(outputs, axis=1)
    assert tf.reduce_all(tf.abs(row_sums - 1.0) < 1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_shape": (32, 32), "num_classes": 2},
        {"input_shape": (32, 32, 3), "num_classes": 1},
        {"input_shape": (32, 32, 3), "num_classes": 2, "learning_rate": 0},
        {"input_shape": (32, 32, 3), "num_classes": 2, "dropout_rate": 1.0},
    ],
)
def test_build_cnn_model_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        build_cnn_model(**kwargs)

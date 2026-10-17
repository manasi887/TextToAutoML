import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import services.automl.pipeline as automl_pipeline
from services.automl.persistence import load_model_package
from services.automl.predictor import predict_with_model


def _dataset(problem_type):
    rows = []
    for index in range(120):
        value = index % 12
        segment = "north" if index % 2 else "south"
        if problem_type == "classification":
            target = "approved" if value >= 6 else "declined"
        else:
            target = float(value * 2 + (1 if segment == "north" else 0))
        rows.append({"value": value, "segment": segment, "target": target})
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("problem_type", "expected_problem_type"),
    [
        ("classification", "Binary Classification"),
        ("regression", "Regression"),
    ],
)
def test_trained_model_reloads_and_predicts_in_fresh_process(
    problem_type, expected_problem_type, tmp_path, monkeypatch
):
    model_store = tmp_path / "models"
    model_store.mkdir()
    real_save_model_package = automl_pipeline.save_model_package
    monkeypatch.setattr(
        automl_pipeline,
        "save_model_package",
        lambda *args, **kwargs: real_save_model_package(
            *args, storage_dir=model_store, **kwargs
        ),
    )
    monkeypatch.setattr(
        automl_pipeline,
        "tune_selected_model",
        lambda model_name, model, *args, **kwargs: {
            "model": model,
            "status": "Skipped",
            "errors": [],
        },
    )

    training = automl_pipeline.run_automl_pipeline(
        _dataset(problem_type),
        target_column="target",
        problem_type=expected_problem_type,
        random_state=17,
    )

    assert training["status"] == "success"
    model_id = training["model"]["model_id"]
    assert (model_store / f"{model_id}.joblib").is_file()

    package = load_model_package(model_id, storage_dir=model_store)
    assert package["problem_type"] == expected_problem_type
    if problem_type == "classification":
        assert package["preprocessing"]["target_classes"] == ["approved", "declined"]
    records = [
        {"segment": "unseen", "value": 2, "extra": "ignored"},
        {"segment": "north", "value": 10},
        {"value": None, "segment": None},
    ]
    expected = predict_with_model(package, records)["predictions"]

    with pytest.raises(ValueError, match="missing required feature column"):
        predict_with_model(package, [{"value": 2}])
    with pytest.raises(ValueError, match="must be a list"):
        predict_with_model(package, {"value": 2})
    with pytest.raises(ValueError, match="must be an object"):
        predict_with_model(package, [None])

    backend_dir = Path(__file__).resolve().parents[1] / "backend"
    script = """
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from services.automl.persistence import load_model_package
from services.automl.predictor import predict_with_model
package = load_model_package(sys.argv[1], storage_dir=Path(sys.argv[2]))
def fail_fit(*args, **kwargs):
    raise AssertionError("prediction must not fit or retrain")
package["model"].fit = fail_fit
for encoder_info in package["encoders"].values():
    encoder = encoder_info.get("encoder")
    if encoder is not None:
        encoder.fit = fail_fit
records = json.loads(sys.argv[4])
print(json.dumps(predict_with_model(package, records)))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            model_id,
            str(model_store),
            str(backend_dir),
            json.dumps(records),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    after_reload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert after_reload["model_id"] == model_id
    assert after_reload["problem_type"] == expected_problem_type
    assert after_reload["predictions"] == pytest.approx(expected)


def test_tabular_loader_rejects_image_model_id_and_corrupt_artifact(tmp_path):
    model_store = tmp_path / "models"
    model_store.mkdir()
    image_model_id = "img_20250101_000000_abcdef"
    image_package_dir = model_store / image_model_id
    image_package_dir.mkdir()
    (image_package_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (image_package_dir / "model.pt").write_bytes(b"image weights")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_model_package(image_model_id, storage_dir=model_store)

    corrupt_model_id = "corrupt_model"
    (model_store / f"{corrupt_model_id}.joblib").write_bytes(b"not a joblib package")
    with pytest.raises(Exception):
        load_model_package(corrupt_model_id, storage_dir=model_store)

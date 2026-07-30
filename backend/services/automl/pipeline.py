from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from services.automl.evaluator import (
    evaluate_models,
    select_best_model,
)
from services.automl.problem_detection import (
    detect_problem_type,
    detect_target_candidates,
)
from services.automl.trainer import (
    prepare_training_data,
    save_trained_model,
    train_models,
)


def run_automl_pipeline(df: pd.DataFrame) -> Dict[str, object]:
    """
    Run the full AutoML pipeline from target selection through model persistence.

    Architecture:
    - Detect the strongest supervised target candidate.
    - Infer the problem type from the selected target.
    - Prepare training data with encoding and imputation.
    - Train a small suite of candidate models.
    - Evaluate the models on the available dataset.
    - Select the best model according to the task-specific metric.
    - Persist the selected model to disk.

    The implementation remains modular by delegating each step to a helper
    function instead of embedding all logic in one large block.
    """

    target_candidates = detect_target_candidates(df).get("target_candidates", [])
    selected_target = target_candidates[0]["column"] if target_candidates else None

    if selected_target is None:
        return {
            "problem_type": "Clustering",
            "selected_target": None,
            "best_model": None,
            "evaluation": {"results": []},
            "saved_model": {
                "status": "Skipped",
                "reason": "No supervised target candidate was detected."
            },
        }

    problem_report = detect_problem_type(df, selected_target)
    problem_type = problem_report["problem_type"]

    preparation = prepare_training_data(df, selected_target)
    X_train = preparation["X"]
    y_train = preparation["y"]

    trained_models = train_models(X_train, y_train, problem_type)["trained_models"]
    evaluation = evaluate_models(trained_models, X_train, y_train, problem_type)
    best_model_report = select_best_model(evaluation["results"], problem_type)

    saved_model = save_trained_model(
        trained_models[best_model_report["best_model"]],
        best_model_report["best_model"],
    )

    return {
        "problem_type": problem_type,
        "selected_target": selected_target,
        "best_model": best_model_report["best_model"],
        "evaluation": evaluation,
        "saved_model": saved_model,
    }

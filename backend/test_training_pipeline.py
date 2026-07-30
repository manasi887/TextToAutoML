import json
import sys
from pathlib import Path

import pandas as pd

# Ensure backend imports work when running from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.dataset.loader import load_dataset
from services.dataset.validator import validate_dataset_df
from services.automl.pipeline import run_automl_pipeline


def main():
    csv_path = Path("storage/uploads/SampleSuperstore.csv")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Sample dataset not found at {csv_path}. "
            "Please place SampleSuperstore.csv in backend/storage/uploads/."
        )

    df = load_dataset(str(csv_path))
    validation = validate_dataset_df(df)

    print("=== AutoML Training Pipeline Demo ===")
    print(f"Dataset: {csv_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print()

    print("=== Validation ===")
    print(json.dumps(validation, indent=2))
    print()

    if not validation["valid"]:
        print("Dataset is not valid. Aborting AutoML pipeline.")
        return

    pipeline_report = run_automl_pipeline(df)

    print("=== Detected Target ===")
    print(pipeline_report.get("selected_target"))
    print()

    print("=== Problem Type ===")
    print(pipeline_report.get("problem_type"))
    print()

    print("=== Models Trained ===")
    evaluation_results = pipeline_report.get("evaluation", {}).get("results", [])
    for result in evaluation_results:
        print(f"- {result['model']}")
    print()

    print("=== Evaluation Metrics ===")
    print(json.dumps(evaluation_results, indent=2))
    print()

    print("=== Best Model ===")
    print(pipeline_report.get("best_model"))
    print()

    print("=== Saved Model ===")
    print(json.dumps(pipeline_report.get("saved_model", {}), indent=2))
    print()


if __name__ == "__main__":
    main()

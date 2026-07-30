import json
import sys
from pathlib import Path

# Ensure backend imports work when running from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.dataset.analyze import analyze_dataset_df
from services.dataset.loader import load_dataset
from services.dataset.preprocess import preprocess_dataset
from services.dataset.validator import validate_dataset_df
from services.automl.problem_detection import generate_automl_recommendation


def main():
    csv_path = Path("storage/uploads/SampleSuperstore.csv")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Sample dataset not found at {csv_path}. "
            "Please place SampleSuperstore.csv in backend/storage/uploads/."
        )

    df = load_dataset(str(csv_path))

    print("=== AutoML Pipeline Debug Script ===")
    print(f"Dataset path: {csv_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print()

    validation_report = validate_dataset_df(df)
    analysis_report = analyze_dataset_df(df)
    processed_df, preprocessing_report = preprocess_dataset(df)
    automl_report = generate_automl_recommendation(processed_df)
    dataset_intelligence_report = automl_report.get("dataset_intelligence", {})

    print("=== Validation ===")
    print(json.dumps(validation_report, indent=2))
    print()

    print("=== Analysis ===")
    print(json.dumps(analysis_report, indent=2))
    print()

    print("=== Preprocessing Report ===")
    print(json.dumps(preprocessing_report, indent=2))
    print()

    print("=== Dataset Intelligence ===")
    print(json.dumps(dataset_intelligence_report, indent=2))
    print()

    print("=== Target Candidates ===")
    print(json.dumps(automl_report.get("target_candidates", []), indent=2))
    print()

    print("=== Detected Problem Type ===")
    print(json.dumps({
        "problem_type": automl_report.get("problem_type"),
        "reason": automl_report.get("problem_detection", {}).get("reason"),
    }, indent=2))
    print()

    print("=== Recommended Models ===")
    print(json.dumps(automl_report.get("recommended_models", []), indent=2))
    print()

    print("=== AutoML Recommendation ===")
    print(json.dumps(automl_report, indent=2))
    print()


if __name__ == "__main__":
    main()

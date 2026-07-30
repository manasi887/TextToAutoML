import json
import sys
from pathlib import Path

import pandas as pd

# Ensure backend imports work when running from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.dataset.intelligence import generate_dataset_recommendations
from services.dataset.loader import load_dataset
from services.dataset.preprocess import preprocess_dataset


def main():
    csv_path = Path("storage/uploads/SampleSuperstore.csv")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Sample dataset not found at {csv_path}. "
            "Please place SampleSuperstore.csv in backend/storage/uploads/."
        )

    df = load_dataset(str(csv_path))
    processed_df, preprocess_report = preprocess_dataset(df)
    intelligence_report = generate_dataset_recommendations(processed_df)

    print("=== Dataset Intelligence Report ===")
    print(f"Rows after preprocessing: {len(processed_df)}")
    print(f"Columns after preprocessing: {len(processed_df.columns)}")
    print()

    print("1. Identifier columns")
    print(json.dumps(intelligence_report["identifier_columns"], indent=2))
    print()

    print("2. Constant columns")
    print(json.dumps(intelligence_report["constant_columns"], indent=2))
    print()

    print("3. High-cardinality columns")
    print(json.dumps(intelligence_report["high_cardinality_columns"], indent=2))
    print()

    print("4. Generated recommendations")
    for index, recommendation in enumerate(intelligence_report["recommendations"], start=1):
        print(f"{index}. {recommendation}")
    print()

    print("=== Preprocessing Report Summary ===")
    print(json.dumps(preprocess_report, indent=2))


if __name__ == "__main__":
    main()

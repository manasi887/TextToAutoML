from pathlib import Path
import sys

# Ensure backend package imports work when the script runs from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    processed_df, report = preprocess_dataset(df)

    original_columns = set(df.columns)
    engineered_columns = [
        column for column in processed_df.columns
        if column not in original_columns
    ]

    print("New columns created:")
    print(engineered_columns)
    print("\nData types of engineered columns:")
    print(processed_df[engineered_columns].dtypes)
    print("\nFirst five rows of engineered features:")
    print(processed_df[engineered_columns].head(5))
    print("\nPreprocessing report:")
    print(report)


if __name__ == "__main__":
    main()

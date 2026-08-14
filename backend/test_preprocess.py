from pathlib import Path

from services.dataset.loader import load_dataset
from services.dataset.preprocess import preprocess_dataset

DATASET_PATH = Path(__file__).resolve().parent / "storage" / "uploads" / "SampleSuperstore.csv"
df = load_dataset(str(DATASET_PATH))

clean_df, report = preprocess_dataset(df)

print("Preprocessing Report:")
print(report)

print("\nData Types:")
print(clean_df.dtypes)

print("\nFinal Shape:", clean_df.shape)
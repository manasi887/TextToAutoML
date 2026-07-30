from services.dataset.loader import load_dataset
from services.dataset.preprocess import preprocess_dataset

df = load_dataset("storage/uploads/SampleSuperstore.csv")

clean_df, report = preprocess_dataset(df)

print("Preprocessing Report:")
print(report)

print("\nData Types:")
print(clean_df.dtypes)

print("\nFinal Shape:", clean_df.shape)
from services.dataset.loader import load_dataset
import pandas as pd


def analyze_dataset_df(df: pd.DataFrame):
    """
    Analyze a DataFrame and return basic information.
    """

    return {
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns)
        },

        "quality": {
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        },

        "columns": df.columns.tolist(),

        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        }
    }


def analyze_dataset(file_path: str):
    """
    Analyze a CSV or Excel dataset and return basic information.
    """

    df = load_dataset(file_path)

    return analyze_dataset_df(df)


from pathlib import Path
import pandas as pd


def load_dataset(file_path: str):
    """
    Load a CSV or Excel dataset and return a Pandas DataFrame.
    """

    file_extension = Path(file_path).suffix.lower()

    if file_extension == ".csv":
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin1")

    elif file_extension == ".xlsx":
        df = pd.read_excel(file_path)

    else:
        raise ValueError("Unsupported file format.")

    return df
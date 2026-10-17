
from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "education_career_success.csv"
)

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    return df

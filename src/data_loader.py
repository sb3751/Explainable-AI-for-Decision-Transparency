# src/data_loader.py

import pandas as pd
from src.config import RAW_DATA_DIR

DATA_FILE = RAW_DATA_DIR / "credit_default.csv"


def load_credit_data() -> pd.DataFrame:
    """
    Load and clean the UCI Credit Card Default dataset.
    Handles:
    - Extra header rows
    - CSV export artifacts
    - Type coercion
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. "
            "Please place credit_default.csv in data/raw/"
        )

    df = pd.read_csv(DATA_FILE)

    # Drop accidental index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop duplicated header row if present
    if isinstance(df.loc[0, "Y"], str):
        df = df.iloc[1:].reset_index(drop=True)

    # Enforce numeric types
    df = df.apply(pd.to_numeric, errors="raise")

    return df


def get_feature_target_split(df: pd.DataFrame):
    """
    Split dataframe into features (X) and target (y).
    """
    X = df.drop(columns=["Y"])
    y = df["Y"]

    return X, y


def basic_data_report(df: pd.DataFrame) -> dict:
    """
    Generate dataset diagnostics for Phase 6.1 validation.
    """
    report = {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "feature_columns": list(df.drop(columns=["Y"]).columns),
        "target_column": "Y",
        "target_distribution": df["Y"].value_counts().to_dict(),
        "missing_values": int(df.isnull().sum().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    return report

"""Data loading and preprocessing utilities for ChurnSense AI."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churn"
NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "support_tickets", "late_payments", "monthly_data_gb",
]
CATEGORICAL_FEATURES = ["contract_type", "internet_service"]


def load_dataset(path="data/customer_churn.csv"):
    """Load the customer dataset and validate required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python train.py` to generate it."
        )
    frame = pd.read_csv(path)
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def build_preprocessor():
    """Create a leakage-safe transformer for numeric and categorical fields."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])

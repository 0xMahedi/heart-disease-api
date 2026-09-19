"""Train and persist the heart disease classification model."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.schemas import FEATURE_NAMES


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "heart.csv"
MODEL_PATH: Final[Path] = PROJECT_ROOT / "model" / "heart_model.joblib"
TARGET_NAME: Final[str] = "target"
MISSING_TEXT_VALUES: Final[set[str]] = {"", "?", "NA", "N/A", "nan", "NaN"}


def _coerce_numeric(series: pd.Series, column_name: str) -> pd.Series:
    """Convert a CSV column to numbers while rejecting unexpected text."""

    normalized = series.astype("string").str.strip()
    normalized = normalized.mask(normalized.isin(MISSING_TEXT_VALUES), pd.NA)
    converted = pd.to_numeric(normalized, errors="coerce")
    invalid = converted.isna() & normalized.notna()
    if invalid.any():
        examples = normalized[invalid].drop_duplicates().tolist()[:3]
        raise ValueError(
            f"Column '{column_name}' contains non-numeric values: {examples}. "
            "Clean the CSV before training."
        )
    return converted


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load, validate, and numerically prepare the expected dataset columns."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the Kaggle CSV and save it as data/heart.csv."
        )

    frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"The CSV contains duplicate column names: {duplicates}.")

    required_columns = set(FEATURE_NAMES) | {TARGET_NAME}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"The CSV is missing required columns: {missing_columns}. "
            f"Expected features: {list(FEATURE_NAMES)} and target: {TARGET_NAME}."
        )
    if frame.empty:
        raise ValueError("The dataset is empty.")

    features = pd.DataFrame(
        {
            feature: _coerce_numeric(frame[feature], feature)
            for feature in FEATURE_NAMES
        }
    )
    target = _coerce_numeric(frame[TARGET_NAME], TARGET_NAME)
    if target.isna().any():
        raise ValueError("The target column contains missing values; remove those rows before training.")

    target_values = set(target.unique().tolist())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"The target column must contain only 0 and 1, but found {sorted(target_values)}."
        )
    target = target.astype(int)
    if target.nunique() != 2:
        raise ValueError("The target column must contain both classes: 0 and 1.")
    if target.value_counts().min() < 2:
        raise ValueError("Each target class must contain at least two rows for a stratified split.")

    return features, target


def train_model() -> dict[str, object]:
    """Train the model, print metrics, and save the reusable joblib artifact."""

    features, target = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }

    artifact = {
        "model": pipeline,
        "features": list(FEATURE_NAMES),
        "target": TARGET_NAME,
        "model_type": "LogisticRegression",
        "metrics": metrics,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Dataset: {DATA_PATH}")
    print(f"Rows: {len(features)}")
    print(f"Features: {list(FEATURE_NAMES)}")
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Saved model to: {MODEL_PATH}")
    return artifact


if __name__ == "__main__":
    train_model()

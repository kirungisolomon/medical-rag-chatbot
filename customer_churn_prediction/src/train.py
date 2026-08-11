from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, validate_input


TARGET = "churn"
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "customer_data.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> None:
    data = pd.read_csv(data_path)
    validate_input(data)

    if TARGET not in data.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    X = data.drop(columns=TARGET)
    y = data[TARGET]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
                class_weight="balanced",
            )),
        ]
    )

    pipeline.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)


if __name__ == "__main__":
    train()

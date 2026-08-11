from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from preprocessing import validate_input


TARGET = "churn"
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "customer_data.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"


def evaluate(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict[str, float]:
    data = pd.read_csv(data_path)
    validate_input(data)
    model = joblib.load(model_path)

    X = data.drop(columns=TARGET)
    y = data[TARGET]
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(classification_report(y, predictions, zero_division=0))
    return {"roc_auc": float(roc_auc_score(y, probabilities))}


if __name__ == "__main__":
    print(evaluate())

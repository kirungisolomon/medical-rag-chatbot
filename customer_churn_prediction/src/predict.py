from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from preprocessing import validate_input


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_pipeline.joblib"


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict_customer(customer: dict, path: Path = MODEL_PATH) -> dict[str, float | str]:
    frame = pd.DataFrame([customer])
    validate_input(frame)

    model = load_model(path)
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = "Likely to churn" if probability >= 0.5 else "Likely to stay"

    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
    }

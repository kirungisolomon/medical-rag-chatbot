import pandas as pd
import pytest

from src.preprocessing import build_preprocessor, validate_input


def test_preprocessor_can_transform_valid_input():
    frame = pd.DataFrame(
        [{
            "tenure_months": 12,
            "monthly_charges": 55.0,
            "support_tickets": 2,
            "contract_type": "one_year",
            "payment_method": "credit_card",
            "internet_service": "fiber",
        }]
    )

    validate_input(frame)
    transformed = build_preprocessor().fit_transform(frame)
    assert transformed.shape[0] == 1


def test_validation_rejects_missing_features():
    frame = pd.DataFrame([{"tenure_months": 12}])

    with pytest.raises(ValueError):
        validate_input(frame)

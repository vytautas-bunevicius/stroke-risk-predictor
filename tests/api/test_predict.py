"""Unit tests for the stroke risk prediction API."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.predict import predict_bp  # pylint: disable=wrong-import-position


@pytest.fixture
def client():
    """Creates a Flask test client fixture."""
    app = Flask(__name__)
    app.register_blueprint(predict_bp)
    with app.test_client() as test_client:
        yield test_client


def test_predict_success(client):  # pylint: disable=redefined-outer-name
    """Tests the /predict endpoint with valid input."""
    with patch("src.services.model_service.predict_stroke_risk") as mock_predict:
        mock_predict.return_value = {
            "prediction": 0.5085650479735153,
            "feature_importances": {
                "age": 17.423976268260386,
                "age_glucose": 13.080690691128256,
            },
            "success": True,
        }
        input_data = {
            "age": 65,
            "hypertension": 0,
            "heart_disease": 1,
            "ever_married": "1",
            "residence_type": "1",
            "avg_glucose_level": 100.0,
            "bmi": 28.5,
            "gender": "Female",
            "work_type": "Private",
            "smoking_status": "never smoked",
        }
        response = client.post("/predict", json=input_data)
        assert response.status_code == 200
        assert response.json["success"] is True
        assert "prediction" in response.json
        assert "feature_importances" in response.json["prediction"]
        assert response.json["prediction"]["prediction"] == 0.5085650479735153
        assert response.json["prediction"]["success"] is True


def test_predict_failure(client):  # pylint: disable=redefined-outer-name
    """Tests the /predict endpoint handling an exception."""
    with patch("src.services.model_service.predict_stroke_risk") as mock_predict:
        mock_predict.side_effect = Exception("Test Error")
        response = client.post("/predict", json={"feature1": "invalid"})
        assert response.status_code == 400
        assert response.json["success"] is False
        assert "error" in response.json


def test_features(client):  # pylint: disable=redefined-outer-name
    """Tests the /features endpoint."""
    response = client.get("/features")
    assert response.status_code == 200
    assert set(response.json) >= {"age", "hypertension", "heart_disease"}


if __name__ == "__main__":
    pytest.main()

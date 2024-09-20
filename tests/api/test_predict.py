"""Unit tests for the stroke risk prediction API.

This module contains pytest-based unit tests for the Flask API endpoints
related to stroke risk prediction.

Note:
    This module modifies the Python path to include the project's root directory.
    This ensures that the 'src' module can be imported correctly regardless of
    where the tests are run from.

Functions:
    test_predict_success: Tests the /predict endpoint with valid input.
    test_predict_failure: Tests the /predict endpoint's error handling.
    test_features: Tests the /features endpoint.

Fixtures:
    client: Creates a Flask test client for the application.

Usage:
    pytest tests/api/test_predict.py

Troubleshooting:
    If imports still fail, verify your project structure and ensure you're
    running pytest from the project's root directory.
"""

import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from unittest.mock import patch  # noqa: E402

import pytest
from flask import Flask  # noqa: E402

from src.api.predict import predict_bp


@pytest.fixture
def client():
    """Creates a Flask test client fixture.
    This fixture sets up a Flask application instance, registers the
    predict_bp blueprint, and provides a test client for interacting
    with the application during tests.
    Yields:
        FlaskClient: A test client for the Flask application.
    """
    app = Flask(__name__)
    app.register_blueprint(predict_bp)
    with app.test_client() as client:
        yield client


def test_predict_success(client):
    """Test the /predict endpoint with valid input."""
    with patch("src.services.model_service.predict_stroke_risk") as mock_predict:
        mock_predict.return_value = {
            "prediction": 0.5085650479735153,
            "feature_importances": {
                "age": 17.423976268260386,
                "age_glucose": 13.080690691128256,
                # ... other feature importances ...
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
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.data.decode()}")
        assert response.status_code == 200
        assert response.json["success"] is True
        assert "prediction" in response.json
        assert "feature_importances" in response.json["prediction"]
        assert response.json["prediction"]["prediction"] == 0.5085650479735153
        assert response.json["prediction"]["success"] is True


def test_predict_failure(client):
    """Test the /predict endpoint handling an exception."""
    with patch("src.services.model_service.predict_stroke_risk") as mock_predict:
        mock_predict.side_effect = Exception("Test Error")
        response = client.post("/predict", json={"feature1": "invalid"})
        assert response.status_code == 400
        assert response.json["success"] is False
        assert "error" in response.json


def test_features(client):
    """Test the /features endpoint."""
    response = client.get("/features")
    assert response.status_code == 200
    assert set(response.json) >= set(["age", "hypertension", "heart_disease"])
    # We're using a subset check here because the actual API seems to return more features


if __name__ == "__main__":
    pytest.main()

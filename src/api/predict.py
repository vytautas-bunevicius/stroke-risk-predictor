"""API endpoints for stroke risk prediction."""

from flask import Blueprint, request, jsonify
from src.services.model_service import predict_stroke_risk, get_input_features

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """Endpoint to predict stroke risk.

    Returns:
        A JSON object containing the prediction result and status.
    """
    data = request.json
    try:
        prediction = predict_stroke_risk(data)
        return (
            jsonify(
                {
                    "success": True,
                    "prediction": prediction,
                    "message": "Prediction successful",
                }
            ),
            200,
        )
    except Exception as e:  # pylint: disable=broad-except
        print(e)
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "message": "Error occurred during prediction",
                }
            ),
            400,
        )


@predict_bp.route("/features", methods=["GET"])
def features():
    """Endpoint to get input features.

    Returns:
        A JSON array of input feature names.
    """
    return jsonify(get_input_features())

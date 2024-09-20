"""Service module for stroke risk prediction model."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "catboost_final_model.joblib"
)
FEATURE_NAMES_PATH = (
    Path(__file__).parent.parent.parent / "models" / "feature_names.joblib"
)

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)


def predict_stroke_risk(features: Dict[str, Any]) -> Dict[str, Any]:
    """Predicts stroke risk based on input features.

    Args:
        features: A dictionary of feature names and values.

    Returns:
        A dictionary containing prediction results and feature importances.

    Raises:
        ValueError: If a required feature is missing.
    """
    logger.info("Received features for prediction: %s", features)

    required_features = [
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "residence_type",
        "bmi",
        "gender",
        "smoking_status",
    ]

    for feature in required_features:
        if feature not in features:
            raise ValueError(f"Missing required feature: {feature}")

    try:
        features["avg_glucose_level"] = features.get("glucose_level", 0)

        # Calculate derived features
        features["age_glucose"] = features["age"] * features["avg_glucose_level"]
        features["age_hypertension"] = features["age"] * features["hypertension"]
        features["age_heart_disease"] = features["age"] * features["heart_disease"]
        features["age_squared"] = features["age"] ** 2
        features["glucose_squared"] = features["avg_glucose_level"] ** 2
        features["bmi_age"] = features["bmi"] * features["age"]
        features["bmi_glucose"] = features["bmi"] * features["avg_glucose_level"]

        # Create one-hot encoded features for categorical variables
        for feature in ["gender", "smoking_status"]:
            features[f"{feature}_{features[feature]}"] = 1

        feature_vector = [features.get(feature, 0) for feature in feature_names]
        logger.info("Processed feature vector: %s", feature_vector)

        prediction = model.predict_proba([feature_vector])[0][1]
        logger.info("Model prediction: %s", prediction)

        feature_importances = dict(zip(feature_names, model.feature_importances_))

        return {
            "success": True,
            "prediction": float(prediction),
            "feature_importances": feature_importances,
        }

    except Exception as e: # pylint: disable=broad-exception-caught
        logger.exception("Error in predict_stroke_risk: %s", str(e))
        return {"success": False, "error": str(e)}


def get_input_features() -> List[str]:
    """Returns the list of input features required for prediction."""
    return [
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "residence_type",
        "bmi",
        "gender",
        "smoking_status",
        "glucose_level",
    ]

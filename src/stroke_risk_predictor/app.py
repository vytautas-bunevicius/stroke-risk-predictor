"""Main application module for the stroke risk predictor."""

from flask import Flask, render_template
from stroke_risk_predictor.api.prediction_endpoints import predict_bp

app = Flask(__name__, template_folder="../../templates", static_folder="../../static")

app.register_blueprint(predict_bp, url_prefix="/api")


@app.route("/", methods=["GET"])
def index():
    """Renders the main page of the application."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

"""Main application module for the stroke risk predictor."""

from flask import Flask, render_template
from pathlib import Path

from stroke_risk_predictor.api.prediction_endpoints import predict_bp

_PACKAGE_ROOT = Path(__file__).parent
_UI_ROOT = _PACKAGE_ROOT / "ui"

app = Flask(
    __name__,
    template_folder=str(_UI_ROOT / "templates"),
    static_folder=str(_UI_ROOT / "static"),
)

app.register_blueprint(predict_bp, url_prefix="/api")


@app.route("/", methods=["GET"])
def index():
    """Renders the main page of the application."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

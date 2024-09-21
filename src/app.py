"""Main application module for the stroke risk predictor."""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from flask import Flask, render_template  # pylint: disable=wrong-import-position
from src.api.predict import predict_bp  # pylint: disable=wrong-import-position

app = Flask(__name__, template_folder="../templates", static_folder="../static")

app.register_blueprint(predict_bp, url_prefix="/api")

@app.route("/", methods=["GET"])
def index():
    """Renders the main page of the application."""
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

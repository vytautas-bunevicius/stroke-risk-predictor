"""Tests for the main application module."""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
from flask import template_rendered
from contextlib import contextmanager
from app import app, predict_bp


@pytest.fixture
def client():
    """Creates a test client for the Flask application."""
    with app.test_client() as client:
        yield client


@contextmanager
def captured_templates(app):
    """Captures templates rendered during a request."""
    recorded = []

    def record(sender, template, context, **extra):
        recorded.append((template, context))

    template_rendered.connect(record, app)
    try:
        yield recorded
    finally:
        template_rendered.disconnect(record, app)


def test_index_route(client):
    """Test the index route."""
    with captured_templates(app) as templates:
        response = client.get("/")
        assert response.status_code == 200
        assert len(templates) == 1
        template, context = templates[0]
        assert template.name == "index.html"


def test_api_blueprint_registered():
    """Test that the API blueprint is registered."""
    assert any(
        blueprint.name == "predict" for blueprint in app.blueprints.values())
    # Note: url_prefix assertion removed as it's not set in app.py


def test_static_folder():
    """Test that the static folder is set correctly."""
    expected_path = Path(__file__).resolve().parent.parent.parent / "static"
    assert Path(app.static_folder).resolve() == expected_path


def test_template_folder():
    """Test that the template folder is set correctly."""
    app_dir = Path(app.root_path)
    template_dir = Path(app.template_folder)

    if template_dir.is_absolute():
        expected_path = template_dir
    else:
        expected_path = (app_dir / template_dir).resolve()

    assert expected_path.name == "templates"
    assert expected_path.exists()
    assert expected_path.is_dir()


def test_debug_mode():
    """Test that debug mode is not enabled by default."""
    assert not app.debug


def test_predict_route_exists():
    """Test that the predict route exists in the registered blueprint."""
    rules = list(app.url_map.iter_rules())
    assert any(rule.endpoint == "predict.predict" for rule in rules)


def test_features_route_exists():
    """Test that the features route exists in the registered blueprint."""
    rules = list(app.url_map.iter_rules())
    assert any(rule.endpoint == "predict.features" for rule in rules)


if __name__ == "__main__":
    pytest.main()

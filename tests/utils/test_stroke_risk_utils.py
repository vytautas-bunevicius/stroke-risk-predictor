"""Tests for stroke_risk_utils.py."""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from plotly.graph_objs import Figure

from src.utils.stroke_risk_utils import (
    plot_combined_histograms,
    plot_combined_bar_charts,
    plot_combined_boxplots,
    plot_correlation_matrix,
    detect_anomalies_iqr,
    flag_anomalies,
    calculate_cramers_v,
    evaluate_model,
    plot_model_performance,
    plot_combined_confusion_matrices,
    extract_feature_importances,
    plot_feature_importances,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "bmi": [20, 22, 24, 26, 28],
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "stroke": [0, 0, 1, 0, 1],
        }
    )


def test_plot_combined_histograms(sample_df):
    """Test if plot_combined_histograms function returns a Figure object."""
    fig = plot_combined_histograms(sample_df, ["age", "bmi"])
    assert isinstance(fig, Figure)


def test_plot_combined_bar_charts(sample_df):
    """Test if plot_combined_bar_charts function returns a Figure object."""
    fig = plot_combined_bar_charts(sample_df, ["gender"])
    assert isinstance(fig, Figure)


def test_plot_combined_boxplots(sample_df):
    """Test if plot_combined_boxplots function returns a Figure object."""
    fig = plot_combined_boxplots(sample_df, ["age", "bmi"])
    assert isinstance(fig, Figure)


def test_plot_correlation_matrix(sample_df):
    """Test if plot_correlation_matrix function calls px.imshow()."""
    with patch("plotly.express.imshow") as mock_imshow:
        plot_correlation_matrix(sample_df, ["age", "bmi"])
        mock_imshow.assert_called_once()


def test_detect_anomalies_iqr(sample_df):
    """Test if detect_anomalies_iqr returns a DataFrame."""
    anomalies = detect_anomalies_iqr(sample_df, ["age", "bmi"])
    assert isinstance(anomalies, pd.DataFrame)
    assert set(anomalies.columns) == {"age", "bmi"}


def test_flag_anomalies(sample_df):
    """Test if flag_anomalies returns a boolean Series."""
    flags = flag_anomalies(sample_df, ["age", "bmi"])
    assert isinstance(flags, pd.Series)
    assert flags.dtype == bool
    assert len(flags) == len(sample_df)


def test_calculate_cramers_v():
    """Test if calculate_cramers_v returns a value between 0 and 1."""
    contingency_table = pd.DataFrame([[10, 20], [30, 40]])
    v = calculate_cramers_v(contingency_table)
    assert 0 <= v <= 1


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.predict_proba.return_value = np.array(
        [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]]
    )
    model.predict.return_value = np.array([1, 0, 1, 0, 1])
    return model


def test_evaluate_model(mock_model, sample_df):
    """Test if evaluate_model returns a dict with expected keys."""
    x = sample_df[["age", "bmi"]]
    y = sample_df["stroke"]
    results = evaluate_model(mock_model, x, y)
    assert isinstance(results, dict)
    expected_keys = {
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
    }
    assert expected_keys.issubset(results.keys())


def test_plot_model_performance():
    """Test if plot_model_performance returns a Figure object."""
    results = {
        "Model1": {"accuracy": 0.8, "precision": 0.7},
        "Model2": {"accuracy": 0.75, "precision": 0.8},
    }
    fig = plot_model_performance(results, ["accuracy", "precision"])
    assert isinstance(fig, Figure)


def test_plot_combined_confusion_matrices():
    """Test if plot_combined_confusion_matrices returns a Figure object."""
    results = {"Model1": {"accuracy": 0.8}, "Model2": {"accuracy": 0.75}}
    y_test = np.array([0, 1, 0, 1, 1])
    y_pred_dict = {
        "Model1": np.array([0, 1, 0, 0, 1]),
        "Model2": np.array([0, 1, 1, 0, 1]),
    }
    fig = plot_combined_confusion_matrices(results, y_test, y_pred_dict)
    assert isinstance(fig, Figure)


def test_extract_feature_importances(mock_model, sample_df):
    """Test if extract_feature_importances returns expected array."""
    x = sample_df[["age", "bmi"]]
    y = sample_df["stroke"]
    mock_model.feature_importances_ = np.array([0.6, 0.4])
    importances = extract_feature_importances(mock_model, x, y)
    assert isinstance(importances, np.ndarray)
    assert len(importances) == 2
    np.testing.assert_array_almost_equal(importances, [0.6, 0.4])


def test_plot_feature_importances():
    """Test if plot_feature_importances returns a Figure object."""
    feature_importances = {
        "Model1": {"age": 0.6, "bmi": 0.4},
        "Model2": {"age": 0.5, "bmi": 0.5},
    }
    fig = plot_feature_importances(feature_importances)
    assert isinstance(fig, Figure)


if __name__ == "__main__":
    pytest.main()

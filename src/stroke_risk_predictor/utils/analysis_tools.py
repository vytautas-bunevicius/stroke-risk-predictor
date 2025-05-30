"""Utilities for stroke risk analysis and model evaluation.

This module provides functions for analyzing stroke risk factors and evaluating
machine learning models for stroke prediction. It includes tools for data
visualization, statistical analysis, anomaly detection, and model performance
evaluation.

Functions:
    plot_combined_histograms: Plot histograms for specified features.
    plot_combined_bar_charts: Plot bar charts for categorical features.
    plot_combined_boxplots: Plot boxplots for numerical features.
    plot_correlation_matrix: Plot a correlation matrix for numerical features.
    detect_anomalies_iqr: Detect anomalies using the IQR method.
    flag_anomalies: Flag anomalies in a DataFrame.
    calculate_cramers_v: Calculate Cramer's V for categorical variables.
    evaluate_model: Evaluate a model's performance.
    plot_model_performance: Plot performance metrics for multiple models.
    plot_combined_confusion_matrices: Plot confusion matrices for multiple models.
    extract_feature_importances: Extract feature importances from a model.
    plot_feature_importances: Plot feature importances across different models.

Classes:
    CustomVotingClassifier: Custom voting classifier for ensemble learning.
    CustomLogisticRegressionWrapper: Wrapper for logistic regression with
        custom class weights.

Usage:
    from stroke_risk_predictor.utils import analysis_tools

    # Plot histograms of risk factors
    analysis_tools.plot_combined_histograms(df, ['age', 'bmi'], nbins=30)

    # Evaluate stroke prediction model
    results = analysis_tools.evaluate_model(model, x_test, y_test, 'Test Set')

    # Create custom ensemble
    ensemble = analysis_tools.CustomVotingClassifier(estimators=[...])

Note:
    This module uses a specific color scheme for visualizations, customizable
    via global color variables.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


BACKGROUND_COLOR = "#EEECE2"
PRIMARY_COLORS = ["#CC7B5C", "#D4A27F", "#EBDBBC", "#9C8AA5"]
SECONDARY_COLORS = [
    "#91A694",
    "#8B9BAE",
    "#666663",
    "#BFBFBA",
    "#E5E4DF",
    "#F0F0EB",
    "#FAFAF7",
]
ALL_COLORS = PRIMARY_COLORS + SECONDARY_COLORS


def plot_combined_histograms(
    df: pd.DataFrame,
    features: List[str],
    nbins: int = 40,
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Plots combined histograms for specified features in the DataFrame.

    Args:
        df: DataFrame containing the features to plot.
        features: List of feature names to plot histograms for.
        nbins: Number of bins for each histogram. Defaults to 40.
        save_path: Optional path to save the plot image.

    Returns:
        The plotly Figure object containing the histograms.
    """
    title = f"Distribution of {', '.join(features)}"
    rows, cols = 1, len(features)

    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.1)

    axis_font = {"family": "Styrene A", "color": "#191919"}

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Histogram(
                x=df[feature],
                nbinsx=nbins,
                name=feature,
                marker={
                    "color": PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                    "line": {"color": "#000000", "width": 1},
                },
            ),
            row=1,
            col=i + 1,
        )

        fig.update_xaxes(
            title_text=feature,
            row=1,
            col=i + 1,
            title_standoff=25,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
        )
        fig.update_yaxes(
            title_text="Count",
            row=1,
            col=i + 1,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
        )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font={"family": "Styrene B", "size": 20, "color": "#191919"},
        showlegend=False,
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        height=500,
        width=400 * len(features),
        margin={"l": 50, "r": 50, "t": 80, "b": 80},
        font={**axis_font, "size": 12},
    )

    if save_path:
        fig.write_image(save_path)

    return fig


def plot_combined_bar_charts(
    df: pd.DataFrame,
    features: List[str],
    max_features_per_plot: int = 3,
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Plots combined bar charts for specified categorical features in the
    DataFrame.

    Args:
        df: DataFrame containing the features to plot.
        features: List of categorical feature names to plot bar charts for.
        max_features_per_plot: Maximum number of features to display per plot. Defaults to 3.
        save_path: Optional path to save the plot images.

    Returns:
        The plotly Figure object containing the bar charts.
    """
    feature_chunks = [
        features[i : i + max_features_per_plot]
        for i in range(0, len(features), max_features_per_plot)
    ]

    axis_font = {"family": "Styrene A", "color": "#191919"}

    if not feature_chunks:
        return go.Figure()

    fig = _create_bar_chart_figure(df, feature_chunks[0], axis_font)

    if save_path:
        for chunk_index, chunk in enumerate(feature_chunks):
            chunk_fig = (
                fig
                if chunk_index == 0
                else _create_bar_chart_figure(df, chunk, axis_font)
            )
            chunk_fig.write_image(f"{save_path}_chunk_{chunk_index + 1}.png")

    return fig


def _create_bar_chart_figure(
    df: pd.DataFrame, feature_chunk: List[str], axis_font: Dict[str, str]
) -> go.Figure:
    """Helper function to create a bar chart figure for a chunk of features.

    Args:
        df: DataFrame containing the data.
        feature_chunk: List of features to plot.
        axis_font: Font configuration for axes.

    Returns:
        The plotly Figure object.
    """
    title = f"Distribution of {', '.join(feature_chunk)}"
    rows, cols = 1, len(feature_chunk)

    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.1)

    for i, feature in enumerate(feature_chunk):
        value_counts = df[feature].value_counts().reset_index()
        value_counts.columns = [feature, "count"]
        fig.add_trace(
            go.Bar(
                x=value_counts[feature],
                y=value_counts["count"],
                name=feature,
                marker={
                    "color": PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                    "line": {"color": "#000000", "width": 1},
                },
            ),
            row=1,
            col=i + 1,
        )
        fig.update_xaxes(
            title_text=feature,
            row=1,
            col=i + 1,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
            showticklabels=True,
        )
        fig.update_yaxes(
            title_text="Count",
            row=1,
            col=i + 1,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
        )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font={"family": "Styrene B", "size": 20, "color": "#191919"},
        showlegend=False,
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        height=500,
        width=400 * len(feature_chunk),
        margin={"l": 50, "r": 50, "t": 80, "b": 150},
        font={**axis_font, "size": 12},
    )

    return fig


def plot_combined_boxplots(
    df: pd.DataFrame, features: List[str], save_path: Optional[str] = None
) -> go.Figure:
    """
    Plots combined boxplots for specified numerical features in the DataFrame.

    Args:
        df: DataFrame containing the features to plot.
        features: List of numerical feature names to plot boxplots for.
        save_path: Optional path to save the plot image.

    Returns:
        The plotly Figure object containing the boxplots.
    """
    title = f"Boxplots of {', '.join(features)}"
    rows, cols = 1, len(features)

    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.1)

    axis_font = {"family": "Styrene A", "color": "#191919"}

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Box(
                y=df[feature],
                marker={
                    "color": PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                    "line": {"color": "#000000", "width": 1},
                },
                boxmean="sd",
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )
        fig.update_yaxes(
            title_text="Value",
            row=1,
            col=i + 1,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
        )
        fig.update_xaxes(
            tickvals=[0],
            ticktext=[feature],
            row=1,
            col=i + 1,
            title_font={**axis_font, "size": 14},
            tickfont={**axis_font, "size": 12},
            showticklabels=True,
        )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font={"family": "Styrene B", "size": 20, "color": "#191919"},
        showlegend=False,
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        height=500,
        width=400 * len(features),
        margin={"l": 50, "r": 50, "t": 80, "b": 150},
        font={**axis_font, "size": 12},
    )

    if save_path:
        fig.write_image(save_path)

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame, numerical_features: List[str], save_path: Optional[str] = None
) -> go.Figure:
    """Plots the correlation matrix of the specified numerical features.

    Args:
        df: DataFrame containing the data.
        numerical_features: List of numerical features to include in the
            correlation matrix.
        save_path: Optional path to save the image file.

    Returns:
        The plotly Figure object containing the correlation matrix.
    """
    numerical_df = df[numerical_features]
    correlation_matrix = numerical_df.corr()
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale=PRIMARY_COLORS,
        title="Correlation Matrix",
    )

    fig.update_layout(
        title={
            "text": "Correlation Matrix",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        title_font=dict(size=24),
        template="plotly_white",
        height=800,
        width=800,
        margin=dict(l=100, r=100, t=100, b=100),
        xaxis=dict(tickangle=-45, title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
    )

    if save_path:
        fig.write_image(save_path)

    return fig


def detect_anomalies_iqr(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Detects anomalies in multiple features using the IQR method.

    Args:
        df: DataFrame containing the data.
        features: List of features to detect anomalies in.

    Returns:
        DataFrame containing the anomalies for each feature.
    """
    anomalies_list = []

    for feature in features:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame.")
            continue

        if not np.issubdtype(df[feature].dtype, np.number):
            print(f"Feature '{feature}' is not numerical and will be skipped.")
            continue

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        feature_anomalies = df[
            (df[feature] < lower_bound) | (df[feature] > upper_bound)
        ]
        if not feature_anomalies.empty:
            print(f"Anomalies detected in feature '{feature}':")
            print(feature_anomalies)
        else:
            print(f"No anomalies detected in feature '{feature}'.")
        anomalies_list.append(feature_anomalies)

    if anomalies_list:
        anomalies = pd.concat(anomalies_list).drop_duplicates().reset_index(drop=True)
        anomalies = anomalies[features]
    else:
        anomalies = pd.DataFrame(columns=features)

    return anomalies


def flag_anomalies(df: pd.DataFrame, features: List[str]) -> pd.Series:
    """Flag anomalies in a DataFrame based on the IQR method.

    Identifies anomalies in specified features using the Interquartile Range
    (IQR) method.

    Args:
        df: The input DataFrame containing the data.
        features: A list of column names in the DataFrame to check for
            anomalies.

    Returns:
        A Series of boolean values where True indicates an anomaly in any of
        the specified features.
    """
    anomaly_flags = pd.Series(False, index=df.index)

    for feature in features:
        first_quartile = df[feature].quantile(0.25)
        third_quartile = df[feature].quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range

        feature_anomalies = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        anomaly_flags |= feature_anomalies

    return anomaly_flags


def calculate_cramers_v(contingency_table: pd.DataFrame) -> float:
    """Calculates Cramer's V for a given contingency table.

    Args:
        contingency_table: A contingency table of categorical variables.

    Returns:
        The calculated Cramer's V value.
    """
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    return cramers_v


def evaluate_model(
    model,
    x: np.ndarray,
    y: np.ndarray,
    dataset_name: Optional[str] = None,
    threshold: Optional[float] = None,
    target_recall: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate a model's performance with optional threshold adjustment.

    Args:
        model: The trained model to evaluate.
        x: Features array.
        y: True labels array.
        dataset_name: Name of the dataset for display purposes.
        threshold: Custom threshold for classification.
        target_recall: Target recall for threshold adjustment.

    Returns:
        Dictionary containing various performance metrics including ROC AUC,
        PR AUC, F1 score, precision, recall, balanced accuracy, threshold,
        predictions, and prediction probabilities.
    """
    y_pred_proba = model.predict_proba(x)[:, 1]

    if target_recall is not None:
        _, recalls, thresholds = precision_recall_curve(y, y_pred_proba)
        idx = np.argmin(np.abs(recalls - target_recall))
        threshold = thresholds[idx]
        print(f"Adjusted threshold: {threshold:.4f}")

    if threshold is not None:
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(x)

    if dataset_name:
        print(f"\nResults on {dataset_name} set:")

    print(classification_report(y, y_pred, zero_division=1))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print(f"ROC AUC: {roc_auc_score(y, y_pred_proba):.4f}")
    print(f"PR AUC: {average_precision_score(y, y_pred_proba):.4f}")
    print(f"F1 Score: {f1_score(y, y_pred, zero_division=1):.4f}")
    print(f"Precision: {precision_score(y, y_pred, zero_division=1):.4f}")
    print(f"Recall: {recall_score(y, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y, y_pred):.4f}")

    return {
        "roc_auc": roc_auc_score(y, y_pred_proba),
        "pr_auc": average_precision_score(y, y_pred_proba),
        "f1": f1_score(y, y_pred, zero_division=1),
        "precision": precision_score(y, y_pred, zero_division=1),
        "recall": recall_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "threshold": threshold if threshold is not None else 0.5,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def plot_model_performance(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Plots and optionally saves a bar chart of model performance metrics with
    legend on the right.

    Args:
        results: A dictionary with model names as keys and dicts of performance metrics as values.
        metrics: List of performance metrics to plot (e.g., 'Accuracy', 'Precision').
        save_path: Optional path to save the plot image.

    Returns:
        The plotly Figure object containing the performance metrics.
    """
    model_names = list(results.keys())
    data = {
        metric: [results[name][metric] for name in model_names] for metric in metrics
    }

    fig = go.Figure()

    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=data[metric],
                name=metric,
                marker=dict(color=ALL_COLORS[i % len(ALL_COLORS)]),
                text=[f"{value:.2f}" for value in data[metric]],
                textposition="auto",
            )
        )

    axis_font = {"family": "Styrene A", "color": "#191919"}

    fig.update_layout(
        barmode="group",
        title={
            "text": "Comparison of Model Performance Metrics",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"family": "Styrene B", "size": 24, "color": "#191919"},
        },
        xaxis_title="Model",
        yaxis_title="Value",
        legend_title="Metrics",
        font={**axis_font, "size": 14},
        height=500,
        width=1200,
        template="plotly_white",
        legend={"yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02},
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
    )

    fig.update_yaxes(range=[0, 1], showgrid=True, gridwidth=1, gridcolor="LightGrey")
    fig.update_xaxes(tickangle=-45, tickfont={**axis_font, "size": 12})

    if save_path:
        fig.write_image(save_path)

    return fig


def plot_combined_confusion_matrices(
    results: Dict[str, Dict[str, float]],
    y_test: np.ndarray,
    y_pred_dict: Dict[str, np.ndarray],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Plot confusion matrices for multiple models in a single figure.

    This function creates a combined plot of confusion matrices for multiple
    models, allowing for easy comparison of model performance. It uses a
    heatmap representation with color coding and percentage annotations.

    Args:
        results: A dictionary where keys are model names and values are
            dictionaries containing model performance metrics.
        y_test: True labels of the test set.
        y_pred_dict: A dictionary where keys are model names and values are
            arrays of predicted labels.
        labels: Optional custom labels for the confusion matrix axes. If None,
            default labels ["No Stroke", "Stroke"] will be used.
        save_path: Optional file path to save the plot as an image.

    Returns:
        The plotly Figure object containing the confusion matrices.

    Raises:
        ValueError: If the number of models in results and y_pred_dict don't match.

    Note:
        This function uses plotly for visualization and assumes binary
        classification (e.g., stroke prediction). The plot is styled with
        predefined color schemes and fonts.
    """
    n_models = len(results)

    if n_models <= 2:
        rows, cols = 1, 2
    else:
        rows, cols = 2, 2

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(results.keys()) + [""] * (rows * cols - n_models),
        vertical_spacing=0.2,
        horizontal_spacing=0.1,
    )

    axis_font = {"family": "Styrene A", "color": "#191919"}

    for i, (name, _) in enumerate(results.items()):
        row = i // cols + 1
        col = i % cols + 1

        cm = confusion_matrix(y_test, y_pred_dict[name])
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        text = [
            [
                f"TN: {cm[0][0]}<br>({cm_percent[0][0]:.1f}%)",
                f"FP: {cm[0][1]}<br>({cm_percent[0][1]:.1f}%)",
            ],
            [
                f"FN: {cm[1][0]}<br>({cm_percent[1][0]:.1f}%)",
                f"TP: {cm[1][1]}<br>({cm_percent[1][1]:.1f}%)",
            ],
        ]

        colorscale = [
            [0, PRIMARY_COLORS[2]],
            [0.33, PRIMARY_COLORS[1]],
            [0.66, PRIMARY_COLORS[1]],
            [1, PRIMARY_COLORS[0]],
        ]

        heatmap = go.Heatmap(
            z=cm,
            x=labels or ["No Stroke", "Stroke"],
            y=labels or ["No Stroke", "Stroke"],
            hoverongaps=False,
            text=text,
            texttemplate="%{text}",
            colorscale=colorscale,
            showscale=False,
        )

        fig.add_trace(heatmap, row=row, col=col)

        fig.update_xaxes(
            title_text="Predicted",
            row=row,
            col=col,
            tickfont={**axis_font, "size": 10},
            title_standoff=25,
        )
        fig.update_yaxes(
            title_text="Actual",
            row=row,
            col=col,
            tickfont={**axis_font, "size": 10},
            title_standoff=25,
        )

    height = 600 if n_models <= 2 else 1000
    width = 1200

    fig.update_layout(
        title_text="Confusion Matrices for All Models",
        title_x=0.5,
        title_font={"family": "Styrene B", "size": 24, "color": "#191919"},
        height=height,
        width=width,
        showlegend=False,
        font={**axis_font, "size": 12},
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        margin=dict(t=100, b=50, l=50, r=50),
    )

    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=16, family="Styrene B", color="#191919")
        i["y"] = i["y"] + 0.03

    if save_path:
        fig.write_image(save_path)

    return fig


def extract_feature_importances(model, x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Extract feature importances using permutation importance for models that do
    not directly provide them.

    Args:
        model: Trained model.
        x: Feature data (DataFrame).
        y: Target data (Series or array).

    Returns:
        Array of feature importances.
    """
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    else:
        perm_import = permutation_importance(model, x, y, n_repeats=30, random_state=42)
        return perm_import.importances_mean


def plot_feature_importances(
    feature_importances: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Plots and optionally saves a bar chart of feature importances across
    different models.

    Args:
        feature_importances: A dictionary with model names
        as keys and dicts of feature importances as values.
        save_path: Optional path to save the plot image.

    Returns:
        The plotly Figure object containing the feature importances.
    """
    fig = go.Figure()

    axis_font = {"family": "Styrene A", "color": "#191919"}

    for i, (name, importances) in enumerate(feature_importances.items()):
        fig.add_trace(
            go.Bar(
                x=list(importances.keys()),
                y=list(importances.values()),
                name=name,
                marker=dict(color=PRIMARY_COLORS[i % len(PRIMARY_COLORS)]),
                text=[f"{value:.3f}" for value in importances.values()],
                textposition="auto",
            )
        )

    fig.update_layout(
        title={
            "text": "Feature Importances Across Models",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"family": "Styrene B", "size": 24, "color": "#191919"},
        },
        xaxis_title="Features",
        yaxis_title="Importance",
        barmode="group",
        template="plotly_white",
        legend_title="Models",
        font={**axis_font, "size": 14},
        height=600,
        width=1200,
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
    )

    fig.update_xaxes(tickangle=-45, tickfont={**axis_font, "size": 12})
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        tickfont={**axis_font, "size": 12},
    )

    if save_path:
        fig.write_image(save_path)

    return fig


class CustomVotingClassifier(VotingClassifier):
    """Custom implementation of VotingClassifier for ensemble learning.

    This class extends sklearn's VotingClassifier to provide customized
    behavior for supervised ensemble learning tasks where models are
    combined for predictive purposes.
    """

    def fit(self, x, y, *, sample_weight=None, **fit_params):
        """Fit the voting classifier.

        Args:
            x: Training input samples of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
            sample_weight: Individual weights for each sample. If None,
                all samples are assumed to have equal weight.
            **fit_params: Additional parameters passed to the estimators.

        Returns:
            self: Returns the instance itself.
        """
        return super().fit(x, y, sample_weight=sample_weight, **fit_params)


class CustomLogisticRegressionWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for logistic regression models with custom class weights.

    This class wraps logistic regression models to add functionalities
    such as handling class weights. It allows for seamless integration
    of custom class weights during the fitting process.
    """

    def __init__(self, model, class_weight):
        """Initialize the wrapper.

        Args:
            model: The logistic regression model to wrap.
            class_weight: Dictionary where keys are class labels and
                values are weights that determine the relative importance
                of each class during model training.
        """
        self.model = model
        self.class_weight = class_weight

    def fit(self, x, y, sample_weight=None):
        """Fit the wrapped model with adjusted sample weights.

        Args:
            x: Features to train the model, typically a 2D array where
                rows correspond to samples and columns to features.
            y: Target labels corresponding to the samples in x.
            sample_weight: Weight assigned to every sample to balance
                their importance during fitting. If not provided,
                uniform weights are used.

        Returns:
            self: Returns the instance itself.
        """
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        sample_weight *= np.array([self.class_weight[yi] for yi in y])
        return self.model.fit(x, y, sample_weight=sample_weight)

    def predict(self, x):
        """Predict class labels for samples in x.

        Args:
            x: The input data for which the predictions are to be made.

        Returns:
            Predicted class labels for the input samples.
        """
        return self.model.predict(x)

    def predict_proba(self, x):
        """Predict class probabilities for samples in x.

        Args:
            x: The input data for which to predict class probabilities.

        Returns:
            Class probabilities of the input samples.
        """
        return self.model.predict_proba(x)

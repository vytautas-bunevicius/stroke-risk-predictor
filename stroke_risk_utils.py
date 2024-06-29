import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.stats import chi2_contingency
from scipy import stats
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
from sklearn.inspection import permutation_importance


PRIMARY_COLORS = ["#5684F7", "#3A5CED", "#7E7AE6"]
SECONDARY_COLORS = ["#7BC0FF", "#B8CCF4", "#18407F", "#85A2FF", "#C2A9FF", "#3D3270"]
ALL_COLORS = PRIMARY_COLORS + SECONDARY_COLORS


def plot_combined_histograms(
    df: pd.DataFrame, features: List[str], nbins: int = 40, save_path: str = None
) -> None:
    """Plots combined histograms for specified features in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to plot.
        features (List[str]): List of features to plot histograms for.
        nbins (int): Number of bins to use in histograms.
        save_path (str): Path to save the image file (optional).
    """
    title = f"Distribution of {', '.join(features)}"
    rows = 1
    cols = len(features)

    fig = sp.make_subplots(
        rows=rows, cols=cols, subplot_titles=features, horizontal_spacing=0.1
    )

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Histogram(
                x=df[feature],
                nbinsx=nbins,
                name=feature,
                marker=dict(
                    color=PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                    line=dict(color="#000000", width=1),
                ),
            ),
            row=1,
            col=i + 1,
        )
        fig.update_xaxes(title_text=feature, row=1, col=i + 1, title_font=dict(size=14))
        fig.update_yaxes(title_text="Count", row=1, col=i + 1, title_font=dict(size=14))

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font=dict(size=20),
        showlegend=False,
        template="plotly_white",
        height=500,
        width=400 * len(features),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.show()

    if save_path:
        fig.write_image(save_path)


def plot_combined_bar_charts(
    df: pd.DataFrame,
    features: List[str],
    max_features_per_plot: int = 3,
    save_path: str = None,
) -> None:
    """Plots combined bar charts for specified categorical features in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to plot.
        features (List[str]): List of categorical features to plot bar charts for.
        max_features_per_plot (int): Maximum number of features to display per plot.
        save_path (str): Path to save the image file (optional).
    """
    feature_chunks = [
        features[i : i + max_features_per_plot]
        for i in range(0, len(features), max_features_per_plot)
    ]

    for chunk_index, feature_chunk in enumerate(feature_chunks):
        title = f"Distribution of {', '.join(feature_chunk)}"
        rows = 1
        cols = len(feature_chunk)

        fig = sp.make_subplots(
            rows=rows, cols=cols, subplot_titles=[None] * cols, horizontal_spacing=0.1
        )

        for i, feature in enumerate(feature_chunk):
            value_counts = df[feature].value_counts().reset_index()
            value_counts.columns = [feature, "count"]
            fig.add_trace(
                go.Bar(
                    x=value_counts[feature],
                    y=value_counts["count"],
                    name=feature,
                    marker=dict(
                        color=PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                        line=dict(color="#000000", width=1),
                    ),
                ),
                row=1,
                col=i + 1,
            )
            fig.update_xaxes(
                title_text=feature,
                row=1,
                col=i + 1,
                title_font=dict(size=14),
                showticklabels=True,
            )
            fig.update_yaxes(
                title_text="Count", row=1, col=i + 1, title_font=dict(size=14)
            )

        fig.update_layout(
            title_text=title,
            title_x=0.5,
            title_font=dict(size=20),
            showlegend=False,
            template="plotly_white",
            height=500,
            width=400 * len(feature_chunk),
            margin=dict(l=50, r=50, t=80, b=150),
        )

        fig.show()

        if save_path:
            file_path = f"{save_path}_chunk_{chunk_index + 1}.png"
            fig.write_image(file_path)


def plot_combined_boxplots(
    df: pd.DataFrame, features: List[str], save_path: str = None
) -> None:
    """Plots combined boxplots for specified numerical features in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to plot.
        features (List[str]): List of numerical features to plot boxplots for.
        save_path (str): Path to save the image file (optional).
    """
    title = f"Boxplots of {', '.join(features)}"
    rows = 1
    cols = len(features)

    fig = sp.make_subplots(
        rows=rows, cols=cols, subplot_titles=[None] * cols, horizontal_spacing=0.1
    )

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Box(
                y=df[feature],
                marker=dict(
                    color=PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                    line=dict(color="#000000", width=1),
                ),
                boxmean="sd",
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )
        fig.update_yaxes(title_text="Value", row=1, col=i + 1, title_font=dict(size=14))
        fig.update_xaxes(
            tickvals=[0],
            ticktext=[feature],
            row=1,
            col=i + 1,
            title_font=dict(size=14),
            showticklabels=True,
        )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font=dict(size=20),
        showlegend=False,
        template="plotly_white",
        height=500,
        width=400 * len(features),
        margin=dict(l=50, r=50, t=80, b=150),
    )

    fig.show()

    if save_path:
        fig.write_image(save_path)


def plot_correlation_matrix(
    df: pd.DataFrame, numerical_features: List[str], save_path: str = None
) -> None:
    """Plots the correlation matrix of the specified numerical features in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        numerical_features (List[str]): List of numerical features to include in the correlation matrix.
        save_path (str): Path to save the image file (optional).
    """
    numerical_df = df[numerical_features]
    correlation_matrix = numerical_df.corr()

    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale=ALL_COLORS,
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

    fig.show()

    if save_path:
        fig.write_image(save_path)


def detect_anomalies_iqr(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Detects anomalies in multiple features using the IQR method.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        features (List[str]): List of features to detect anomalies in.

    Returns:
        pd.DataFrame: DataFrame containing the anomalies for each feature.
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


def flag_anomalies(df, features):
    """
    Identify and flag anomalies in a DataFrame based on the Interquartile Range (IQR) method for specified features.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        features (list of str): A list of column names in the DataFrame to check for anomalies.

    Returns:
        pd.Series: A Series of boolean values where True indicates an anomaly in any of the specified features.
    """
    anomaly_flags = pd.Series(False, index=df.index)

    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        feature_anomalies = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        anomaly_flags |= feature_anomalies

    return anomaly_flags


def calculate_cramers_v(contingency_table):
    """
    Calculates Cramer's V for a given contingency table.

    Args:
        contingency_table (pandas.DataFrame): A contingency table of categorical variables.

    Returns:
        float: The calculated Cramer's V value.
    """
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    return cramers_v


def evaluate_model(model, X, y, dataset_name=None):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    if dataset_name:
        print(f"\nResults on {dataset_name} set:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print(f"ROC AUC: {roc_auc_score(y, y_pred_proba):.4f}")
    print(f"PR AUC: {average_precision_score(y, y_pred_proba):.4f}")
    print(f"F1 Score: {f1_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall: {recall_score(y, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y, y_pred):.4f}")

    return {
        "roc_auc": roc_auc_score(y, y_pred_proba),
        "pr_auc": average_precision_score(y, y_pred_proba),
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
    }


def plot_model_performance(
    results: Dict[str, Dict[str, float]], metrics: List[str], save_path: str = None
) -> None:
    """
    Plots and optionally saves a bar chart of model performance metrics with legend on the right.

    Args:
        results: A dictionary with model names as keys and dicts of performance metrics as values.
        metrics: List of performance metrics to plot (e.g., 'Accuracy', 'Precision').
        save_path: Path to save the image file (optional).
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
                marker_color=ALL_COLORS[i % len(ALL_COLORS)],
                text=[f"{value:.2f}" for value in data[metric]],
                textposition="auto",
            )
        )

    fig.update_layout(
        barmode="group",
        title={
            "text": "Comparison of Model Performance Metrics",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=24),
        },
        xaxis_title="Model",
        yaxis_title="Value",
        legend_title="Metrics",
        font=dict(size=14),
        height=500,
        width=1200,
        template="plotly_white",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02),
    )

    fig.update_yaxes(range=[0, 1], showgrid=True, gridwidth=1, gridcolor="LightGrey")
    fig.update_xaxes(tickangle=-45)

    fig.show()

    if save_path:
        fig.write_image(save_path)


def plot_combined_confusion_matrices(
    results, y_test, y_pred_dict, labels=None, save_path=None
):
    """
    Plots a combined confusion matrix for multiple models.

    Parameters:
    results (dict): A dictionary containing the results of multiple models.
        Each key is the name of a model, and the value is the result of that model.
    y_test (numpy.ndarray): The true labels for the dataset.
    y_pred_dict (dict): A dictionary containing the predicted labels for each model.
        Each key is the name of a model, and the value is the predicted labels for that model.
    labels (list, optional): A list of class labels. If not provided, default labels are used.
    save_path (str, optional): The path to save the image file. If not provided, the image is not saved.

    Returns:
    None
    """
    n_models = len(results)
    if n_models > 4:
        print("Warning: Only the first 4 models will be plotted.")
        n_models = 4

    fig = make_subplots(rows=2, cols=2, subplot_titles=list(results.keys())[:n_models])

    for i, (name, model_results) in enumerate(list(results.items())[:n_models]):
        row = i // 2 + 1
        col = i % 2 + 1

        cm = confusion_matrix(y_test, y_pred_dict[name])
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create custom text for each cell
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

        # Define colorscale with normalized values
        colorscale = [
            [0, ALL_COLORS[2]],  # TN
            [0.33, ALL_COLORS[1]],  # FP
            [0.66, ALL_COLORS[1]],  # FN
            [1, ALL_COLORS[0]],  # TP
        ]

        heatmap = go.Heatmap(
            z=cm,
            x=labels if labels else ["Class 0", "Class 1"],
            y=labels if labels else ["Class 0", "Class 1"],
            hoverongaps=False,
            text=text,
            texttemplate="%{text}",
            colorscale=colorscale,
            showscale=False,
        )

        fig.add_trace(heatmap, row=row, col=col)

        fig.update_xaxes(
            title_text="Predicted", row=row, col=col, tickfont=dict(size=10)
        )
        fig.update_yaxes(title_text="Actual", row=row, col=col, tickfont=dict(size=10))

    fig.update_layout(
        title_text="Confusion Matrices for All Models",
        title_x=0.5,
        height=500,
        width=1200,
        showlegend=False,
        font=dict(size=12),
    )

    fig.show()

    if save_path:
        fig.write_image(save_path)


def extract_feature_importances(model, X, y):
    """
    Extract feature importances using permutation importance for models that do not directly provide them.

    Args:
        model: Trained model
        X: Feature data (DataFrame)
        y: Target data (Series or array)

    Returns:
        Array of feature importances
    """
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    else:
        # Calculate permutation importance
        perm_import = permutation_importance(model, X, y, n_repeats=30, random_state=42)
        return perm_import.importances_mean


def plot_feature_importances(
    feature_importances: Dict[str, Dict[str, float]], save_path: str = None
) -> None:
    """
    Plots and optionally saves a bar chart of feature importances across different models.

    Args:
        feature_importances: A dictionary with model names as keys and dicts of feature importances as values.
        save_path: Path to save the image file (optional).
    """
    fig = go.Figure()

    for i, (name, importances) in enumerate(feature_importances.items()):
        fig.add_trace(
            go.Bar(
                x=list(importances.keys()),
                y=list(importances.values()),
                name=name,
                marker_color=ALL_COLORS[i % len(ALL_COLORS)],
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
            "font": dict(size=24),
        },
        xaxis_title="Features",
        yaxis_title="Importance",
        barmode="group",
        template="plotly_white",
        legend_title="Models",
        font=dict(size=14),
        height=600,
        width=1200,
    )

    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")

    fig.show()

    if save_path:
        fig.write_image(save_path)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.stats import chi2_contingency
from scipy import stats
from typing import Tuple, List
import numpy as np

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

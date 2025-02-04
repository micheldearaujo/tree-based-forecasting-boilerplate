# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')

import logging
from typing import Any, Optional
import yaml
import datetime as dt
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import shap
import xgboost as xgb

from src.utils import *
# from src.config import *
from src.configuration.config_data import *
from src.configuration.config_model import *
from src.configuration.config_feature import *
from src.configuration.config_viz import *

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)


TARGET_COL          = model_config['target_col']
PREDICTED_COL       = model_config['predicted_col']
FORECAST_HORIZON    = model_config['forecast_horizon']


def plot_learning_curve(model, evals_result, model_type):
    """
    Plota as learning curves de XGBoost, LightGBM ou CatBoost usando Plotly.
    
    :param model: O modelo treinado
    :param evals_result: Dicionário contendo os resultados de avaliação (train/val loss)
    :param model_type: Nome do modelo ("xgboost", "lgbm" ou "catboost")
    """
    fig = go.Figure()
    
    if model_type == "XGBRegressor":
        train_metric = list(evals_result['validation_0'].keys())[0]
        val_metric = list(evals_result['validation_1'].keys())[0] if 'validation_1' in evals_result else None
        
        fig.add_trace(go.Scatter(y=evals_result['validation_0'][train_metric],
                                 mode='lines',
                                 name=f'Training - {train_metric}'))
        
        if val_metric:
            fig.add_trace(go.Scatter(y=evals_result['validation_1'][val_metric],
                                     mode='lines',
                                     name=f'Validation - {val_metric}'))
    
    elif model_type == "LGBMRegressor":
        for data_name, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                fig.add_trace(go.Scatter(y=values,
                                         mode='lines',
                                         name=f'{data_name} - {metric_name}'))
    
    elif model_type == "CatBoostRegressor":
        for metric_name, values in evals_result.items():
            fig.add_trace(go.Scatter(y=values["RMSE"],
                                     mode='lines',
                                     name=f'Training - {metric_name}'))
    
    fig.update_layout(title=f"{model_type}'s Learning Curve",
                      xaxis_title="N Iteration",
                      yaxis_title="Metric value",
                      template="plotly_dark")
    return fig


def extract_learning_curves(model: xgb.sklearn.XGBRegressor, display: bool=False) -> matplotlib.figure.Figure:
    """
    Extracting the XGBoost Learning Curves.
    Can display the figure or not.

    Args:
        model (xgb.sklearn.XGBRegressor): Fitted XGBoost model
        display (bool, optional): Display the figure. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Learning curves figure
    """

    logger.debug("Plotting the learning curves...")

    # extract the learning curves
    learning_results = model.evals_result()

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    plt.suptitle("XGBoost Learning Curves")
    axs[0].plot(learning_results['validation_0']['rmse'], label='Training')
    axs[0].set_title("RMSE Metric")
    axs[0].set_ylabel("RMSE")
    axs[0].set_xlabel("Iterations")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(learning_results['validation_0']['logloss'], label='Training')
    axs[1].set_title("Logloss Metric")
    axs[1].set_ylabel("Logloss")
    axs[1].set_xlabel("Iterations")
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()

    if display:
        plt.show()
    
    return fig


def plot_crossval_results(pred_df: pd.DataFrame, model_type: str) -> go.Figure:
    """
    Cria visualizações dos resultados da validação do modelo, comparando valores reais, previsões e erros.

    Args:
    -----------
    pred_df : pd.DataFrame

        DataFrame contendo as colunas 'date', 'ACTUAL', 'FORECAST', 'MAPE' e 'RMSE', onde:
        - 'date': Data das observações.
        - 'ACTUAL': Valores reais.
        - 'FORECAST': Valores previstos pelo modelo.
        - 'MAPE': Erro percentual absoluto médio para cada observação.
        - 'RMSE': Raiz do erro quadrático médio para cada observação.

    model_type : str
        Tipo do modelo utilizado (ex: 'Linear Regression', 'Random Forest').

    Returns:
    --------
    Figure
        Um objeto `plotly.graph_objects.Figure` contendo o gráfico gerado.
    """
    # Calcular métricas médias
    model_mape = pred_df['MAPE'].mean()
    model_rmse = pred_df['RMSE'].mean()

    # Criar o gráfico
    fig = go.Figure()

    # Adicionar linha para valores reais
    fig.add_trace(
        go.Scatter(
            x=pred_df.drop_duplicates(subset=["date"])["date"],
            y=pred_df.drop_duplicates(subset=["date"])["ACTUAL"],
            mode='lines+markers',
            name='Observed values',
            line=dict(color='grey', dash='dash'),
            marker=dict(symbol='hexagon-open', size=8)
        )
    )

    # Definir cores consistentes para cada modelo
    model_colors = {model: color for model, color in zip(pred_df['MODEL_TYPE'].unique(), px.colors.qualitative.Plotly)}

    # Adicionar linhas para previsões de cada modelo, quebrando por janela
    for model_type in pred_df['MODEL_TYPE'].unique():
        model_df = pred_df[pred_df['MODEL_TYPE'] == model_type]
        for window_number in model_df['WINDOW'].unique():
            window_df = model_df[model_df['WINDOW'] == window_number]

            showlegend= True if window_number == model_df['WINDOW'].unique()[0] else False
            fig.add_trace(
                go.Scatter(
                    x=window_df['date'],
                    y=window_df['FORECAST'],
                    mode='lines+markers',
                    name=model_type,
                    line=dict(color=model_colors[model_type]),
                    marker=dict(size=10),
                    showlegend=showlegend  # Mostrar legenda apenas para a primeira janela
                )
            )

    # Personalizar layout
    model = "all models" if pred_df['MODEL_TYPE'].nunique() > 1 else model_type
    fig.update_layout(
        title=f"Cross-validation results for {model}<br>Avg MAPE: {model_mape * 100:.2f}% | Avg RMSE: {model_rmse:.2f}",
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_dark',
        # legend=dict(x=0.02, y=0.98),  # Posicionar legenda no canto superior esquerdo
        legend_title="Model",
        xaxis=dict(tickformat='%Y-%m-%d', tickangle=20)
    )

    return fig


def visualize_forecast(pred_df: pd.DataFrame, historical_df: pd.DataFrame, stock_name: str):
    """
    Creates visualizations of the model forecast

    Paramters:
        pred_df: DataFrame with true values, predictions and the date column
        historical_df: DataFrame with historical values

    Returns:
        None
    """

    logger.info("Vizualizing the results...")

    fig, axs = plt.subplots(figsize=(12, 5), dpi = 200)
    # Plot the ACTUALs
    sns.lineplot(
        data=historical_df,
        x="date",
        y="Close",
        label="Historical values",
        ax=axs
    )
    sns.scatterplot(
        data=historical_df,
        x="date",
        y="Close",
        ax=axs,
        size="Close",
        sizes=(80, 80),
        legend=False
    )

    # Plot the FORECASTs
    sns.lineplot(
        data=pred_df,
        x="date",
        y="FORECAST",
        label="FORECAST values",
        ax=axs
    )
    sns.scatterplot(
        data=pred_df,
        x="date",
        y="FORECAST",
        ax=axs,
        size="FORECAST",
        sizes=(80, 80),
        legend=False
    )

    axs.set_title(f"Default XGBoost {model_config['FORECAST_HORIZON']-4} days FORECAST for {stock_name}")
    axs.set_xlabel("date")
    axs.set_ylabel("R$")

    #plt.show()
    return fig

def plot_tree_feature_importance(feature_importance_df: pd.DataFrame, kind: str) -> go.Figure:
    """
    Plota a importância das features usando Plotly, com opções para gráficos de barras ou boxplots.

    Args:
    -----------
    feature_importance_df : pd.DataFrame
        Um DataFrame contendo as colunas 'Feature' e 'Importance', onde 'Feature' representa o nome da feature
        e 'Importance' representa o valor da importância da feature.

    kind : str
        O tipo de gráfico a ser plotado. Pode ser 'bar' para um gráfico de barras ou 'boxplot' para um boxplot.

    Returns:
    --------
    Figure
        Um objeto `plotly.graph_objects.Figure` contendo o gráfico gerado.
    """
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    if feature_importance_df["MODEL_TYPE"].nunique() > 1:
        model_type = "All Models"
    else:
        model_type = feature_importance_df["MODEL_TYPE"].values[0]

    if kind == 'boxplot':
        fig = px.box(feature_importance_df, x='Importance', y='Feature', orientation='h')
    elif kind == 'bar':
        fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h')
    else:
        raise ValueError('Invalid kind parameter. Expected "boxplot" or "bar".')

    fig.update_layout(
        title=f'Feature Importance for {model_type}',
        xaxis_title='Importance',
        yaxis_title='Feature',
        template='plotly_dark'
    )

    return fig


def plot_scatter(df, x_axis, y_axis):

    fig = px.scatter(
        df, 
        x=x_axis, 
        y=y_axis, 
        title=f'{y_axis} vs {x_axis}', 
        # labels={'oi_invoice_price': 'Price ($)', 'oi_invoice_volume': 'Volume'},
        color_discrete_sequence=colors,
        width=800, height=600
    )

    # Show the plot
    fig.show()


def plot_2y_axis_series(df, x_axis, y1_axis, y2_axis):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_axis],
        y=df[y1_axis],
        name=y1_axis,
        mode='lines+markers',
        line=dict(color=colors[0])
    ))

    fig.add_trace(go.Scatter(
        x=df[x_axis],
        y=df[y2_axis],
        name=y2_axis,
        mode='lines+markers',
        yaxis='y2',
        line=dict(color=colors[1])
    ))

    fig.update_layout(
            title={
            'text': f'Comparing the {y1_axis} vs {y2_axis}',
            'font': {'size': 24}
        },
        xaxis_title={
            'text': 'Date',
            'font': {'size': 20}
        },
        yaxis_title={
            'text': y1_axis,
            'font': {'size': 20}
        },
        yaxis2=dict(
                    title={
                'text': y2_axis,
                'font': {'size': 20}
            },
            overlaying='y',
            side='right'
        ),
        legend=dict(
            x=0.1,
            y=1.1,
            traceorder='normal',
            orientation='h',
            font=dict(size=16)
        ),
        height=600,
        width=1200
    )

    # Show plot
    fig.show()


def plot_multiple_lines(dataframe, x_col: str, y_col: list[str], title="Multiple Line Plots", x_label="X-axis", y_label="Y-axis"):
    """
    Plots multiple line plots using Plotly from a DataFrame with separate columns for each line.

    Parameters:
        dataframe (pandas.DataFrame): DataFrame containing the data to plot. Each column (except x_col) represents a line.
        x_col (str): Name of the column to use for the x-axis.
        title (str): Title of the plot. Default is "Multiple Line Plots".
        x_label (str): Label for the x-axis. Default is "X-axis".
        y_label (str): Label for the y-axis. Default is "Y-axis".

    Returns:
        fig: A Plotly figure object (interactive plot).
    """
    # Create a figure
    fig = go.Figure()

    # Add lines for each column (excluding the x_col)
    for idx, col in enumerate(y_col):
        if col != x_col:
            fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[col], mode='lines', name=col, line=dict(color=colors[idx])))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white"
    )

    return fig


def plot_multiple_models_forecast(
    training_dataframe: pd.DataFrame,
    forecast_dataframe: pd.DataFrame,
    x_col: str,
    y_col: list[str],
    title = "Multiple Line Plots",
    x_label = "X-axis",
    y_label = "Y-axis"
) -> Any:

    # Create a figure
    fig = go.Figure()

    past_values_df = training_dataframe.reset_index().copy().tail(6)
    fig.add_trace(
        go.Scatter(
            x=past_values_df[x_col],
            y=past_values_df[y_col[0]],
            mode='lines+markers',
            name="Last observed data",
            line=dict(color='blue', dash='dash'),
            marker=dict(symbol='hexagon-open', size=10)
        )
    )

    # Add lines for each column (excluding the x_col)
    for model_type in forecast_dataframe["MODEL_TYPE"].unique():
        plotting_df = forecast_dataframe[forecast_dataframe["MODEL_TYPE"] == model_type].copy()

        fig.add_trace(
            go.Scatter(
                x=plotting_df[x_col],
                y=plotting_df[y_col[1]],
                mode='lines+markers',
                name=model_type,
                marker=dict(size=10),
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_dark",
        legend_title="Model"
    )

    return fig

    
def plot_1y_axis_series(df, x_axis, yaxis, title):

    fig = go.Figure()
    for i, col in enumerate(yaxis):
        fig.add_trace(go.Scatter(
            x=df[x_axis],
            y=df[col],
            name=col,
            mode='lines+markers',
            line=dict(color=colors[i])
        ))

    fig.update_layout(
            title={
            'text': f'{title} over time',
            'font': {'size': 24}
        },
        xaxis_title={
            'text': 'Date',
            'font': {'size': 20}
        },
        yaxis_title={
            'text': title,
            'font': {'size': 20}
        },
        legend=dict(
            x=0.1,
            y=1.1,
            traceorder='normal',
            orientation='h',
            font=dict(size=16)
        ),
        height=600,
        width=1200
    )

    # Show plot
    fig.show()


def plot_shap_summary(shap_values, dataset):
    """
    Plota o summary plot dos valores SHAP para um dataset de treinamento.
    
    :param model: Modelo treinado
    :param X_train: Conjunto de dados de treinamento
    """
    # model_type = type(model).__name__
    # try:
    #     if model_type in ["Ridge"]:
    #         explainer = shap.LinearExplainer(model, X_train)
    #     elif model_type in ["SVR"]:
    #         explainer = shap.KernelExplainer(model.predict, X_train)
    #     else:
    #         explainer = shap.TreeExplainer(model)
    # except Exception as e:
    #     # raise AttributeError(f"Falha ao inicializar SHAP explainer para {model_type}. Erro: {str(e)}")
    #     fig, ax = plt.subplots()
    
    # shap_values = explainer.shap_values(X_train)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, dataset, show=False)
    # plt.savefig("shap_summary.png")
    
    return fig


def plot_residuals(residuals: list, y_pred:list):

    # Plotting residuals
    plt.figure(figsize=(16, 12))

    # Train residuals vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')

    # Histogram of Train residuals
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')

    # Show plots
    plt.tight_layout()
    fig = plt.gcf()
    # plt.show()
    plt.close()

    return fig
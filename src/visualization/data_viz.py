# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')

import logging
import yaml
import datetime as dt

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance
import xgboost as xgb

from src.utils import *


with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

with open("src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())
    model_config = config['model_config']
    data_config = config['data_config']
    TARGET_COL = model_config['target_col']
    CATEGORY_COL = model_config['category_col']
    PREDICTED_COL = model_config['predicted_col']
    FORECAST_HORIZON = model_config['forecast_horizon']

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

    fig2, axs2 = plt.subplots(figsize=(6, 3))
    plot_importance(model, ax=axs2, importance_type='weight')
    
    if display:
        plt.show()
    
    return fig, fig2

def visualize_validation_results(pred_df: pd.DataFrame, ticker: str):
    """
    Creates visualizations of the model validation

    Paramters:
        pred_df: DataFrame with true values, predictions and the date column
        model_mape: The validation MAPE
        model_mae: The validation MAE
        model_wape: The validation WAPE

    Returns:
        None
    """

    logger.debug("Vizualizing the results...")
    model_mape = pred_df['MAPE'].values[0]
    model_rmse = pred_df['RMSE'].values[0]

    fig, axs = plt.subplots(figsize=(6, 3))

    # Plot the ACTUALs
    sns.lineplot(
        data=pred_df,
        x="DATE",
        y="ACTUAL",
        label="Actuals",
        ax=axs
    )
    sns.scatterplot(
        data=pred_df,
        x="DATE",
        y="ACTUAL",
        ax=axs,
        size="ACTUAL",
        sizes=(80, 80), legend=False
    )

    # Plot the FORECASTs
    sns.lineplot(
        data=pred_df,
        x="DATE",
        y=PREDICTED_COL,
        label="Forecasted",
        ax=axs
    )
    sns.scatterplot(
        data=pred_df,
        x="DATE",
        y=PREDICTED_COL,
        ax=axs,
        size=PREDICTED_COL,
        sizes=(80, 80), legend=False
    )

    axs.set_title(f"Valdation results for {CATEGORY_COL} - {ticker}\nMAPE: {round(model_mape*100, 2)}% | RMSE: {model_rmse}")
    axs.set_xlabel("DATE")
    axs.set_ylabel("R$")
    axs.tick_params(axis='x', rotation=20)
    plt.grid()
    # try:
    #     plt.savefig(f"./reports/figures/XGBoost_predictions_{dt.datetime.now().date()}_{stock_name}.png")

    # except FileNotFoundError:
    #     logger.warning("FORECAST Figure not Saved!")

    #plt.show()
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

    fig, axs = plt.subplots(figsize=(12, 5), dpi = 2000)
    # Plot the ACTUALs
    sns.lineplot(
        data=historical_df,
        x="DATE",
        y="Close",
        label="Historical values",
        ax=axs
    )
    sns.scatterplot(
        data=historical_df,
        x="DATE",
        y="Close",
        ax=axs,
        size="Close",
        sizes=(80, 80),
        legend=False
    )

    # Plot the FORECASTs
    sns.lineplot(
        data=pred_df,
        x="DATE",
        y="FORECAST",
        label="FORECAST values",
        ax=axs
    )
    sns.scatterplot(
        data=pred_df,
        x="DATE",
        y="FORECAST",
        ax=axs,
        size="FORECAST",
        sizes=(80, 80),
        legend=False
    )

    axs.set_title(f"Default XGBoost {model_config['FORECAST_HORIZON']-4} days FORECAST for {stock_name}")
    axs.set_xlabel("DATE")
    axs.set_ylabel("R$")

    #plt.show()
    return fig

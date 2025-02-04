import sys
import os
sys.path.insert(0,'.')

import re
import yaml
import argparse
import logging
import logging.config
from typing import Any
import datetime as dt

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import pmdarima as pm

from pmdarima import auto_arima

import mlflow
from mlflow.tracking import MlflowClient

from src.features.feat_eng import *
from src.models.train import configure_mlflow_experiment, calculate_shap_values_to_df
from src.configuration.config_data import *
from src.configuration.config_model import *
from src.configuration.config_feature import *
from src.configuration.config_viz import *

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)


PROCESSED_DATA_PATH = data_config['paths']['processed_data_path']
PROCESSED_DATA_NAME = data_config['table_names']['processed_table_name']
OUTPUT_DATA_PATH = data_config['paths']['output_data_path']
OUTPUT_DATA_NAME = data_config['table_names']['output_table_name']
MODELS_PATH = data_config['paths']['models_path']
TARGET_COL = model_config['target_col']
PREDICTED_COL = model_config['predicted_col']
FORECAST_HORIZON = model_config['forecast_horizon']

MLRUNS_PATH = model_config['mlflow_runs_path']


def load_latest_model(model_name: str, alias: str) -> tuple:
    """
    Load the latest version of a trained MLflow model.

    Args:
        model_name (str): The name of the MLflow model to loaded.

    Returns:
        tuple: A tuple containing the loaded model, the model's URI, and the source of the model.
    """
    client = MlflowClient()
    configure_mlflow_experiment()
    model_version = client.get_model_version_by_alias(model_name, alias)
    
    source = model_version.source
    model_uri = source.split('/')
    model_uri[-1] = model_name
    model_uri = "/".join(model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    return model, model_uri, source


def load_latest_model_old(model_name: str) -> tuple:
    """
    Load the latest version of a trained MLflow model.

    Args:
        model_name (str): The name of the MLflow model to loaded.

    Returns:
        tuple: A tuple containing the loaded model, the model's URI, and the source of the model.
    """
    client = MlflowClient()
    configure_mlflow_experiment()
    source = client.get_latest_versions(model_name, stages=["None"])[-1].source
    model_uri = source.split('/')
    model_uri[-1] = model_name
    model_uri = "/".join(model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    # Using Alias: Not Working Yet!!
    # https://mlflow.org/docs/latest/model-registry.html
    # model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")

    return model, model_uri, source



def initialize_lag_values(df: pd.DataFrame, features_list: list, target_column: str, future_df: pd.DataFrame):
    """Calculates and sets the initial lag feature value for a given lag and target column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the historical data.
        features_list (list): The modeling features list.
        target_column (str): The name of the target column.
        future_df (pd.DataFrame): DataFrame to store the future (out-of-sample) features.

    Returns:
        pd.DataFrame: The updated future_df with the calculated lag feature.
    """
    for feature in filter(lambda f: "lag" in f, features_list):
        lag_value = int(feature.split("_")[-1])
        future_df.loc[future_df.index.min(), f"{target_column}_lag_{lag_value}"] = df[target_column].iloc[-lag_value]

    return future_df


def initialize_sma_values(df: pd.DataFrame, features_list: list, target_column: str, future_df: pd.DataFrame):
    """Calculates and sets the initial moving average feature value for a given window size and target column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the historical data.
        features_list (list): The modeling features list.
        target_column (str): The name of the target column.
        future_df (pd.DataFrame): DataFrame to store the future (out-of-sample) features.

    Returns:
        pd.DataFrame: The updated future_df with the calculated moving average feature.
    """
    for feature in filter(lambda f: "sma" in f, features_list):

        ma_value = int(feature.split("_")[-1])
        future_df.loc[future_df.index.min(), f"{target_column}_sma_{ma_value}"] = (
            df[target_column].rolling(ma_value).mean().iloc[-1]
        )

    return future_df


def create_future_frame(model_df: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    """
    Creates a Dataframe Index for future dates based on the last training date and forecast horizon.
    """

    last_training_day = model_df['date'].max()
    logger.info(f'Último dia disponível para treinamento: {last_training_day}')
    future_dates = pd.date_range(
        start=last_training_day + pd.DateOffset(days=1),  # Start one day after the last training date
        periods=forecast_horizon,
        freq='W-FRI'
    )
    future_df = pd.DataFrame({"date": future_dates})

    return future_df


def update_target_lag_features(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates lag features in the future DataFrame based on past target values and the current prediction.
    """
    for feature in filter(lambda f: f"{TARGET_COL}_lag" in f, features):
        # print(f'Updating Feature: {feature}')
        lag_value = int(feature.split("_")[-1])
        future_df.loc[day + 1, feature] = past_target_values[-lag_value]
        future_df = future_df.dropna(subset=["date"])

    return future_df


def update_lag_features(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates lag features in the future DataFrame based on past target values and the current prediction.
    """
    for feature in filter(lambda f: "lag" in f, features):
        lag_value = int(feature.split("_")[-1])
        future_df.loc[day + 1, feature] = past_target_values[-lag_value]
        future_df = future_df.dropna(subset=["date"])

    return future_df


def update_sma_features(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates moving average features in the future DataFrame
    based on past target values and the current prediction.
    """
    for feature in filter(lambda f: "sma" in f, features):
        ma_value = int(feature.split("_")[-1])
        last_n_closing_prices = past_target_values[-ma_value:]
        future_df.loc[day + 1, feature] = np.mean(last_n_closing_prices)

    return future_df


def update_target_sma_features(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates moving average features in the future DataFrame
    based on past target values and the current prediction.
    """
    for feature in filter(lambda f: f"{TARGET_COL}_sma" in f, features):
        ma_value = int(feature.split("_")[-1])
        last_n_closing_prices = past_target_values[-ma_value:]
        future_df.loc[day + 1, feature] = np.mean(last_n_closing_prices)

    return future_df


def update_target_ms_features(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates moving sum features in the future DataFrame
    based on past target values and the current prediction.
    """
    for feature in filter(lambda f: f"{TARGET_COL}_ms" in f, features):
        ms_value = int(feature.split("_")[-1])
        last_n_closing_prices = past_target_values[-ms_value:]
        future_df.loc[day + 1, feature] = np.sum(last_n_closing_prices)

    return future_df


def update_target_cummax_features(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates moving average features in the future DataFrame
    based on past target values and the current prediction.
    """
    for feature in filter(lambda f: f"{TARGET_COL}_cum_max" in f, features):
        future_df.loc[day + 1, feature] = np.max(past_target_values[-52:])

    return future_df


def update_target_comparison_lag_features(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates comparison between lag values (Target variable) in the future DataFrame based on past target values and the current prediction.
    """
    for feature in filter(lambda f: f"comparison_{TARGET_COL}_shift_" in f, features):
        # print(f'Updating Feature: {feature}')
        first_lag_value = int(feature.split("_")[-2])
        second_lag_value = int(feature.split("_")[-1])
        future_df.loc[day + 1, feature] = past_target_values[-first_lag_value] / past_target_values[-second_lag_value]
        future_df = future_df.dropna(subset=["date"])

    return future_df


def update_target_comparison_ms_features(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates comparison between moving sum values (Target variable) in the future DataFrame based on past target values and the current prediction.
    """
    for feature in filter(lambda f: f"comparison_{TARGET_COL}_ms_" in f, features):
        logger.debug(f'Updating Feature: {feature}')

        short_window_value = int(feature.split("_")[-2])
        short_window_last_values = past_target_values[-short_window_value:]

        long_window_value = int(feature.split("_")[-1])
        long_window_last_values = past_target_values[-long_window_value:]

        future_df.loc[day + 1, feature] = 100 * (np.sum(short_window_last_values) / np.sum(long_window_last_values))
        future_df = future_df.dropna(subset=["date"])

    return future_df


def update_target_comparison_sma_features(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates comparison between simple moving averages values (Target variable) in the future DataFrame based on past target values and the current prediction.
    """
    for feature in filter(lambda f: f"comparison_{TARGET_COL}_sma_" in f, features):
        logger.debug(f'Updating Feature: {feature}')

        short_window_value = int(feature.split("_")[-2])
        short_window_last_values = past_target_values[-short_window_value:]

        long_window_value = int(feature.split("_")[-1])
        long_window_last_values = past_target_values[-long_window_value:]

        future_df.loc[day + 1, feature] = 100 * (np.mean(short_window_last_values) / np.mean(long_window_last_values))
        future_df = future_df.dropna(subset=["date"])

    return future_df


def update_bollinger_bands(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates Bollinger Band features in the future DataFrame
    based on past target values and the current prediction.
    """
    for feature in filter(lambda f: "bband" in f, features):
        logger.debug(f"Updating feature {feature}")
        ma_value = int(feature.split("_")[-1])
        last_n_closing_prices = past_target_values[-ma_value:]
        middle_bband = np.mean(last_n_closing_prices)
        std_bband = np.std(last_n_closing_prices)
        upper_bband = middle_bband + 2 * std_bband
        lower_bband = middle_bband - 2 * std_bband
        bband_spread = upper_bband - lower_bband
        future_df.loc[day + 1, feature] = bband_spread

    return future_df


def update_rsi(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates RSI features in the future DataFrame
    based on past target values and the current prediction.
    """
    for feature in filter(lambda f: "rsi" in f, features):
        logger.debug(f"Updating feature {feature}")
        rsi_value = int(feature.split("_")[-1]) + 1

        last_n_closing_prices = past_target_values[-rsi_value:]
        df_last_n_past_values = pd.DataFrame(last_n_closing_prices, columns=[TARGET_COL])
        # Calculate the daily changes
        df_last_n_past_values['change'] = df_last_n_past_values[TARGET_COL].diff()
        # Calculate gains and losses
        df_last_n_past_values['gain'] = df_last_n_past_values['change'].apply(lambda x: max(x, 0))
        df_last_n_past_values['loss'] = df_last_n_past_values['change'].apply(lambda x: abs(min(x, 0)))
        # Calculate the average gains and losses
        avg_gain = df_last_n_past_values['gain'].mean()
        avg_loss = df_last_n_past_values['loss'].mean()
        # Calculate the Relative Strength (RS)
        if avg_loss > 0:
            rs = avg_gain/avg_loss
            rsi = (100 - (100 / (1 + rs)))
        else:
            rsi=100
        future_df.loc[day + 1, feature] = rsi
    return future_df


def forecast_exog_features(original_df: pd.DataFrame, future_exog: pd.DataFrame, lags_exog_dict: dict) -> pd.DataFrame:
    """
    Forecasts external features using ARIMA models for specified lags and appends future values to the dataset.

    This function iterates over a dictionary of external features and their respective lags, applying an ARIMA model to forecast 
    the future values for features with a minimum lag below a certain threshold (5 in this case). It then appends the forecasted 
    values to the original dataset.

    Args:
        original_df (pd.DataFrame): The original dataset containing historical data of external features to be used for forecasting.
        future_exog (pd.DataFrame): A DataFrame with the future dates for which external features need to be forecasted.
        lags_exog_dict (dict): A dictionary where the keys are the names of the external features to be forecasted, 
                               and the values are lists containing the lags (integers) associated with each feature.

    Returns:
        pd.DataFrame: The original dataset concatenated with the forecasted values for the external features, 
                      indexed by both historical and future dates.
    """
    full_df = original_df.copy()
    window_train_date_max = full_df.index.max()

    X_future = pd.DataFrame(np.nan, index=future_exog.index, columns=full_df.columns)
    mlflow.autolog(disable=True)

    for exog in lags_exog_dict:
        min_lag = min(lags_exog_dict[exog])
        if min_lag < FORECAST_HORIZON:
            logger.info(f"Forecasting external feature: {exog}")
            model = auto_arima(
                full_df[exog],
                max_p=2,
                max_d=1,
                max_q=2,
                seasonal=False,
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
            )

            forecast = model.predict(n_periods=FORECAST_HORIZON - min_lag)
            X_future[exog] = forecast
    full_df = pd.concat([full_df, X_future])

    return full_df


def make_future_df(forecast_horzion: int, model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a future dataframe for forecasting.

    Parameters:
        forecast_horizon (int): The number of days to forecast into the future.
        model_df (pandas dataframe): The dataframe containing the training data.

    Returns:
        future_df (pandas dataframe): The future dataframe used for forecasting.
    """
    future_df = create_future_frame(model_df.reset_index(), forecast_horzion)
    future_df = future_df.set_index("date")
    future_df = forecast_exog_features(model_df, future_df, lags_exog_dict)

    X_future, y_future = future_df.drop(columns=[TARGET_COL]).copy(), \
                    future_df[TARGET_COL].copy()

    feature_df = build_features(X_future, y_future)
    future_df = feature_df[feature_df['date'] > model_df.index.max()].copy()
    future_df = future_df.reset_index(drop=True)
    
    return future_df


def make_iterative_predictions(model: Any, future_df: pd.DataFrame, past_target_values: list) -> pd.DataFrame:

    """
    Make predictions for the next `forecast_horizon` periods using a Tree-based model.
    
    Parameters:
        model (sklearn model): Scikit-learn tree-based best model to use to perform inferece.
        future_df (pd.DataFrame): The "Feature" DataFrame (X) with future index.
        past_target_values (list): The target variable's historical values to calculate features on.
        
    Returns:
        pd.DataFrame: The future DataFrame with forecasts.
    """

    future_df_feat = future_df.copy()
    shap_values_inference_df = pd.DataFrame()
    shap_values_list = []
    all_features = future_df_feat.columns
    predictions = []
    weeks = [i for i in range(1, 7)]

    FOERCAST_HORIZON = len(future_df_feat)
    LAST_DAY = FOERCAST_HORIZON-1

    if type(model).__name__ == 'ARIMA':
        predictions = model.predict(n_periods=FORECAST_HORIZON).values
        final_shap_values = np.zeros(future_df.drop(columns=["date"]).shape)

    else:
        for day in range(0, FOERCAST_HORIZON):

            X_inference = future_df_feat.drop(columns=["date"]).loc[[day]]
            prediction = model.predict(X_inference)[0]

            predictions.append(prediction)
            past_target_values.append(prediction)

            if day < LAST_DAY:
                future_df_feat = update_target_lag_features(future_df_feat, day, past_target_values, all_features)
                future_df_feat = update_target_sma_features(future_df_feat, day, past_target_values, all_features)
                future_df_feat = update_target_ms_features(future_df_feat, day, past_target_values, all_features)
                future_df_feat = update_bollinger_bands(future_df_feat, day, past_target_values, all_features)
                future_df_feat = update_target_comparison_lag_features(future_df_feat, day, past_target_values, all_features)

            ########### SHAP VALUES #################################
            shap_values, shap_values_df = calculate_shap_values_to_df(model, X_inference)
            shap_values_inference_df = pd.concat([shap_values_inference_df, shap_values_df], axis=0)
            shap_values_list.append(shap_values)

        final_shap_values = np.vstack(shap_values_list)

    future_df_feat[PREDICTED_COL] = predictions
    future_df_feat['WEEK'] = weeks

    shap_values_inference_df['date'] = future_df_feat['date']
    shap_values_inference_df['WEEK'] = weeks

    return future_df_feat, shap_values_inference_df, final_shap_values


def forecast_external_features(lags_exog_dict: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    """
    Forecasts external features using ARIMA models based on specified lags.

    Args:
        lags_exog_dict (dict): Dictionary with feature names as keys and list of lags as values.
        X_train (pd.DataFrame): The training DataFrame containing the features.
        X_test (pd.DataFrame): The testing DataFrame containing the features.
        forecast_horizon (int): The forecast horizon.

    Returns:
        pd.DataFrame: DataFrame containing the predicted values for the external features.
    """
    X_train_exog_preds  = pd.DataFrame(np.nan, index = X_test.index, columns=X_test.columns)

    logger.info("Forecasting the external features...")
    mlflow.autolog(disable=True)
    
    for exog in lags_exog_dict:
        min_lag = min(lags_exog_dict[exog])
        if min_lag < forecast_horizon:
            logger.debug(f"Forecasting external feature: {exog}")
            model = pm.auto_arima(
                X_train.dropna(subset=[exog])[exog],
                max_p=3,
                max_d=1,
                max_q=3,
                seasonal=False, trace=False, error_action='ignore', suppress_warnings=True
            )

            forecast = model.predict(n_periods=FORECAST_HORIZON - min_lag)
            X_train_exog_preds[exog] = forecast

    return X_train_exog_preds
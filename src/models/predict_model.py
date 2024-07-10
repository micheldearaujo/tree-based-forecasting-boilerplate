import sys
import os
sys.path.insert(0,'.')

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

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

from src.features.feat_eng import create_date_features

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)

with open("src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())
    model_config = config['model_config']
    data_config = config['data_config']
    PROCESSED_DATA_PATH = data_config['paths']['processed_data_path']
    PROCESSED_DATA_NAME = data_config['table_names']['processed_table_name']
    OUTPUT_DATA_PATH = data_config['paths']['output_data_path']
    OUTPUT_DATA_NAME = data_config['table_names']['output_table_name']
    MODELS_PATH = data_config['paths']['models_path']
    TARGET_COL = model_config['target_col']
    CATEGORY_COL = model_config['category_col']
    PREDICTED_COL = model_config['predicted_col']
    FORECAST_HORIZON = model_config['forecast_horizon']
    features_list = config['features_list']
    available_models = model_config['available_models']


def load_production_model_sklearn(ticker):
    """
    Loading the Sklearn models saved using the traditional Joblib format.
    """
    model_file_path = os.path.join(MODELS_PATH, ticker, "prod_model.joblib")
    current_prod_model = joblib.load(model_file_path)

    return current_prod_model


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
    for feature in filter(lambda f: "LAG" in f, features_list):
        lag_value = int(feature.split("_")[-1])
        future_df.loc[future_df.index.min(), f"{target_column}_LAG_{lag_value}"] = df[target_column].iloc[-lag_value]

    return future_df


def initialize_ma_values(df: pd.DataFrame, features_list: list, target_column: str, future_df: pd.DataFrame):
    """Calculates and sets the initial moving average feature value for a given window size and target column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the historical data.
        features_list (list): The modeling features list.
        target_column (str): The name of the target column.
        future_df (pd.DataFrame): DataFrame to store the future (out-of-sample) features.

    Returns:
        pd.DataFrame: The updated future_df with the calculated moving average feature.
    """
    for feature in filter(lambda f: "MA" in f, features_list):

        ma_value = int(feature.split("_")[-1])
        future_df.loc[future_df.index.min(), f"{target_column}_MA_{ma_value}"] = (
            df[target_column].rolling(ma_value).mean().iloc[-1]
        )

    return future_df


def create_future_frame(model_df: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    """
    Creates a Dataframe Index for future dates based on the last training date and forecast horizon.
    """

    last_training_day = model_df["DATE"].max()
    future_dates = pd.date_range(
        start=last_training_day + pd.DateOffset(days=1),  # Start one day after the last training date
        periods=forecast_horizon + 1,
        freq='D'
    )
    future_df = pd.DataFrame({"DATE": future_dates})
    future_df[CATEGORY_COL] = model_df[CATEGORY_COL].iloc[0]  # Use the first stock symbol from the model DataFrame

    return future_df


def drop_weekends(future_df):
    """
    Dropout the weekend days of the future DataFrame,
    since the Stock Market doesn't work on weekends.
    """

    future_df = future_df[~future_df["DAY_OF_WEEK"].isin([5, 6])]
    future_df.reset_index(drop=True, inplace=True)

    return future_df


def make_future_df(forecast_horzion: int, model_df: pd.DataFrame, features_list: list) -> pd.DataFrame:
    """
    Create a future dataframe for forecasting.

    Parameters:
        forecast_horizon (int): The number of days to forecast into the future.
        model_df (pandas dataframe): The dataframe containing the training data.

    Returns:
        future_df (pandas dataframe): The future dataframe used for forecasting.
    """
    future_df = create_future_frame(model_df, forecast_horzion)
    future_df = create_date_features(df=future_df)
    future_df = drop_weekends(future_df)
    future_df = initialize_ma_values(model_df, features_list, TARGET_COL, future_df)
    future_df = initialize_lag_values(model_df, features_list, TARGET_COL, future_df)

    return future_df.reindex(columns=["DATE", CATEGORY_COL, *features_list])


def update_lag_features(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates lag features in the future DataFrame based on past target values.
    """
    for feature in filter(lambda f: "LAG" in f, features):
        lag_value = int(feature.split("_")[-1])
        future_df.loc[day + 1, feature] = past_target_values[-lag_value]
        future_df = future_df.dropna(subset=["DATE"])

    return future_df


def update_ma_features(future_df: pd.DataFrame, day: int, past_target_values: list, features: list) -> pd.DataFrame:
    """
    Updates moving average features in the future DataFrame
    based on past target values and the current prediction.
    """
    for feature in filter(lambda f: "MA" in f, features):
        ma_value = int(feature.split("_")[-1])
        last_n_closing_prices = past_target_values[-ma_value:]
        future_df.loc[day + 1, feature] = np.mean(last_n_closing_prices)

    return future_df


def make_iterative_predictions(model: Any, future_df: pd.DataFrame, past_target_values: list) -> pd.DataFrame:

    """
    Make predictions for the next `forecast_horizon` periods using a Tree-based model.
    
    Parameters:
        model (sklearn model): Scikit-learn tree-based best model to use to perform inferece.
        future_df (pd.DataFrame): The "Feature" DataFrame (X) with future index.
        past_target_values (list): The target variable's historical values to calculate the moving averages on.
        
    Returns:
        pd.DataFrame: The future DataFrame with forecasts.
    """

    future_df_feat = future_df.copy()
    all_features = future_df_feat.columns
    predictions = []

    FH_WITHOUT_WEEKENDS = len(future_df_feat)
    LAST_DAY = FH_WITHOUT_WEEKENDS-1

    for day in range(0, FH_WITHOUT_WEEKENDS):

        X_inference = future_df_feat.drop(columns=["DATE", CATEGORY_COL]).loc[[day]]
        prediction = model.predict(X_inference)[0]

        predictions.append(prediction)
        past_target_values.append(prediction)

        if day < LAST_DAY:
            future_df_feat = update_lag_features(future_df_feat, day, past_target_values, all_features)
            future_df_feat = update_ma_features(future_df_feat, day, past_target_values, all_features)
    
    future_df_feat[PREDICTED_COL] = predictions
    future_df_feat[PREDICTED_COL] = future_df_feat[PREDICTED_COL].astype('float64').round(2)

    return future_df_feat


def make_iterative_predictions_ensemble(model: Any, future_df: pd.DataFrame, past_target_values: list) -> pd.DataFrame:

    """
    Make predictions for the next `forecast_horizon` periods using a Tree-based model.
    
    Parameters:
        model (sklearn model): Scikit-learn tree-based best model to use to perform inferece.
        future_df (pd.DataFrame): The "Feature" DataFrame (X) with future index.
        past_target_values (list): The target variable's historical values to calculate the moving averages on.
        
    Returns:
        pd.DataFrame: The future DataFrame with forecasts.
    """

    future_df_feat = future_df.copy()
    all_features = future_df_feat.columns
    predictions = []

    FH_WITHOUT_WEEKENDS = len(future_df_feat)
    LAST_DAY = FH_WITHOUT_WEEKENDS-1

    for day in range(0, FH_WITHOUT_WEEKENDS):

        X_inference = future_df_feat.drop(columns=["DATE", CATEGORY_COL]).loc[[day]]
        prediction = model.predict(X_inference)[0]

        predictions.append(prediction)
        past_target_values.append(prediction)

        if day < LAST_DAY:
            future_df_feat = update_lag_features(future_df_feat, day, past_target_values, all_features)
            future_df_feat = update_ma_features(future_df_feat, day, past_target_values, all_features)
    
    return predictions
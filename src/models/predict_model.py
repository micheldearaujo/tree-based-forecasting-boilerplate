import sys
import os
sys.path.insert(0,'.')

import re
import warnings
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

from src.features.feat_eng import create_date_features

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)

with open("src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())

def load_production_model_sklearn(model_type, ticker_symbol):
    """
    Loading the Sklearn models saved using the traditional Joblib format.
    """
    MODELS_PATH = config['paths']['models_path']
    model_file_path = f"{MODELS_PATH}/{model_type}/Model_{ticker_symbol}.joblib"
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
    future_df["STOCK"] = model_df["STOCK"].iloc[0]  # Use the first stock symbol from the model DataFrame

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
    TARGET_NAME = config['model_config']['TARGET_NAME']

    future_df = create_future_frame(model_df, forecast_horzion)
    future_df = create_date_features(df=future_df)
    future_df = drop_weekends(future_df)
    future_df = initialize_ma_values(model_df, features_list, TARGET_NAME, future_df)
    future_df = initialize_lag_values(model_df, features_list, TARGET_NAME, future_df)

    return future_df.reindex(columns=["DATE", "STOCK", *features_list])


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
    Make predictions for the next `forecast_horizon` days using a XGBoost model
    
    Parameters:
        model (sklearn model): Scikit-learn best model to use to perform inferece.
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

        X_inference = future_df_feat.drop(columns=["DATE", "STOCK"]).loc[[day]]
        prediction = model.predict(X_inference)[0]

        predictions.append(prediction)
        past_target_values.append(prediction)

        if day < LAST_DAY:
            future_df_feat = update_lag_features(future_df_feat, day, past_target_values, all_features)
            future_df_feat = update_ma_features(future_df_feat, day, past_target_values, all_features)
    
    future_df_feat["FORECAST"] = predictions
    future_df_feat["FORECAST"] = future_df_feat["FORECAST"].astype('float64').round(2)

    return future_df_feat



def inference_pipeline(model_type=None, ticker_symbol=None, write_to_table=True):
    """
    Executes the model inference pipeline to generate stock price predictions.

    This function performs the following steps:

    1. Loads the featurized dataset from 'processed_stock_prices.csv'.
    2. Filters the dataset by ticker symbol if provided.
    3. Filters the models to be used for inference based on the provided model type.
    4. For each selected ticker symbol and model type:
        a. Loads the corresponding production model.
        b. Creates a future DataFrame for the forecast horizon.
        c. Generates predictions using the loaded model.
    5. Concatenates all predictions into a single DataFrame.
    6. Optionally writes the predictions to a file or database table.

    Args:
        model_type (str, optional): The type of model to use for inference. If None, all available models in the configuration will be used. Defaults to None.
        ticker_symbol (str, optional): The ticker symbol of the stock to predict. If None, predictions are made for all stocks in the dataset. Defaults to None.
        write_to_table (bool, optional): If True, the predictions will be written to a file or database table. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions for all selected ticker symbols and model types.

    Raises:
        ValueError: If an invalid model type is provided.
    """

    FORECAST_HORIZON = config['model_config']['forecast_horizon']
    available_models = config['model_config']['available_models']
    TARGET_NAME = config['model_config']['TARGET_NAME']
    features_list = config['features_list']
    PROCESSED_DATA_PATH = config['paths']['processed_data_path']
    OUTPUT_DATA_PATH = config['paths']['output_data_path']

    logger.debug("Loading the featurized dataset...")
    stock_df_feat_all = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["DATE"])
    
    final_predictions_df = pd.DataFrame()

    # Check the ticker_symbol parameter
    if ticker_symbol:
        ticker_symbol = ticker_symbol.upper() + '.SA'
        stock_df_feat_all = stock_df_feat_all[stock_df_feat_all["STOCK"] == ticker_symbol]

    # Check the model_type parameter 
    if model_type is not None and model_type not in available_models:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from: {available_models}")
    
    elif model_type:
        available_models = [model_type.upper()]

    for ticker_symbol in stock_df_feat_all["STOCK"].unique():
        stock_df_feat = stock_df_feat_all[stock_df_feat_all["STOCK"] == ticker_symbol].copy()

        for model_type in available_models:
            logger.debug(f"Performing inferece for ticker symbol [{ticker_symbol}] using model [{model_type}]...")
 
            current_prod_model = load_production_model_sklearn(model_type, ticker_symbol)

            logger.debug("Creating the future dataframe...")
            future_df = make_future_df(FORECAST_HORIZON, stock_df_feat, features_list)

            logger.debug("Predicting...")
            predictions_df = make_iterative_predictions(
                model=current_prod_model,
                future_df=future_df,
                past_target_values=list(stock_df_feat[TARGET_NAME].values)
            )
            predictions_df['MODEL_TYPE'] = model_type
            final_predictions_df = pd.concat([final_predictions_df, predictions_df], axis=0)

    # Add the run date
    RUN_DATE = dt.datetime.today().date()
    final_predictions_df["RUN_DATE"] = RUN_DATE

    if write_to_table:
        logger.info("Writing the predictions to database...")

        file_path = f"{OUTPUT_DATA_PATH}/output_stock_prices.csv"
        if os.path.isfile(file_path):
            final_predictions_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            final_predictions_df.to_csv(file_path, index=False)

        logger.info("Predictions written successfully!")

    return final_predictions_df

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform Out-of-Sample Tree-based models Inference.")

    parser.add_argument(
        "-mt", "--model_type",
        type=str,
        choices=["xgb", "et"],
        help="Model name use for inference (xgb, et) (optional, defaults to all)."
    )
    parser.add_argument(
        "-ts", "--ticker_symbol",
        type=str,
        help="""Ticker Symbol for inference. (optional, defaults to all).
        Example: bova11 -> BOVA11.SA | petr4 -> PETR4.SA"""
    )
    parser.add_argument(
        "-w", "--write_to_table",
        action="store_false",
        help="Disable Writing the OFS forecasts to table. Defaults to True. Run '--write_to_table' to Disable."
    )
    args = parser.parse_args()

    logger.info("Starting the Inference pipeline...")
    inference_pipeline(args.model_type.upper(), args.ticker_symbol, args.write_to_table)
    logger.info("Inference Pipeline completed successfully!")
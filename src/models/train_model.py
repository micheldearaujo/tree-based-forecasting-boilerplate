# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

import os
import yaml
import argparse
import logging
import logging.config
import joblib

import pandas as pd

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)

with open("src/configuration/hyperparams.yaml", 'r') as f:  
    hyperparams = yaml.safe_load(f.read())

with open("src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())
    model_config = config['model_config']

def train_model(X_train, y_train, model_type, ticker_symbol, tune_params, save_model):
    """Trains a tree-based regression model."""

    MODELS_PATH = config['paths']['models_path']
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH+f'/{model_type}', exist_ok=True)
    base_params = hyperparams['BASE_PARAMS'][model_type]
    model_file_path = f"{MODELS_PATH}/{model_type}/Model_{ticker_symbol}.joblib"

    if tune_params:
        best_params = tune_params_gridsearch(X_train, y_train, model_type, ticker_symbol)
        base_params.update(best_params)

    if model_type == 'XGB':
        model = xgb.XGBRegressor(objective='reg:squarederror', **base_params).fit(X_train, y_train)

    else:
        model = ExtraTreesRegressor(**base_params).fit(X_train, y_train)

    if save_model:
        joblib.dump(model, model_file_path)

    return model


def tune_params_gridsearch(X: pd.DataFrame, y: pd.Series, model_type:str, ticker_symbol: str, n_splits=3):
    """
    Performs time series hyperparameter tuning on a model using grid search.
    
    Args:
        X (pd.DataFrame): The input feature data
        y (pd.Series): The target values
        model_type (str): The model to tune. Options: ['XGB', 'ET']
        ticker_symbol (str): Ticker Symbol to perform Tuning on.
        n_splits (int): Number of folds for cross-validation (default: 3)
    
    Returns:
        best_params (dict): The best hyperparameters found by the grid search
    """

    logger.info(f"Performing hyperparameter tuning for [{ticker_symbol}] using {model_type}...")

    model = xgb.XGBRegressor() if model_type == "XGB" else ExtraTreesRegressor()

    param_distributions = hyperparams['PARAM_SPACES'][model_type]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid_search = GridSearchCV(
        model,
        param_grid=param_distributions,
        cv=tscv,
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        verbose=1
    ).fit(X, y)
    
    best_params = grid_search.best_params_
    logger.info(f"Best parameters found: {best_params}")

    return best_params


def split_feat_df_Xy(df):
    """Splits the featurized dataframe to train the ML models."""
    X_train=df.drop([model_config["TARGET_NAME"], "DATE"], axis=1)
    y_train=df[model_config["TARGET_NAME"]]

    return X_train, y_train


def training_pipeline(tune_params=False, model_type=None, ticker_symbol=None, save_model=False):
    """
    Executes the complete model training pipeline for one or multiple ticker symbols and model types.

    The pipeline performs the following steps:
    
    1. Loads the featurized dataset from 'processed_stock_prices.csv'.
    2. Filters the dataset based on the specified ticker symbol (if provided).
    3. Trains the specified models (or all available models if none specified).
    4. Optionally performs hyperparameter tuning using RandomizedSearchCV.
    5. Optionally saves the trained models to files.

    Args:
        tune_params (bool, optional): If True, perform hyperparameter tuning. Defaults to False.
        model_type (str, optional): The model type to train. If None, all available models will be trained. Valid choices are defined in `model_config['available_models']`. Defaults to None.
        ticker_symbol (str, optional): The ticker symbol to train on. If None, all ticker symbols in the dataset will be used. Valid choices are defined in `data_config['tickers_list']`.  Defaults to None.
        save_model (bool, optional): If True, save the trained models to files. Defaults to False.

    Raises:
        ValueError: If an invalid model type is provided.
    """
    PROCESSED_DATA_PATH = config['paths']['processed_data_path']


    logger.debug("Loading the featurized dataset..")
    all_ticker_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["DATE"])
    logger.info(f"Last Available Date in Training Dataset: {all_ticker_df['DATE'].max()}")

    # Check the ticker_symbol parameter
    if ticker_symbol:
        ticker_symbol = ticker_symbol.upper() + '.SA'
        all_ticker_df = all_ticker_df[all_ticker_df["STOCK"] == ticker_symbol]

    # Check the model_type parameter
    available_models = model_config['available_models']
    if model_type is not None and model_type not in available_models:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from: {available_models}")
    
    elif model_type:
        available_models = [model_type.upper()]
    
    for ticker_symbol in all_ticker_df["STOCK"].unique():
        ticker_df_feat = all_ticker_df[all_ticker_df["STOCK"] == ticker_symbol].drop("STOCK", axis=1).copy()

        X_train, y_train = split_feat_df_Xy(ticker_df_feat)

        for model_type in available_models:
            logger.debug(f"Training model [{model_type}] for Ticker Symbol [{ticker_symbol}]...")
            xgb_model = train_model(X_train, y_train, model_type, ticker_symbol, tune_params, save_model)

            # learning_curves_fig , feat_importance_fig = extract_learning_curves(xgboost_model)


# Execute the whole pipeline
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Tree-based models with optional hyperparameter tuning.")
    parser.add_argument(
        "-t", "--tune",
        action="store_true",
        help="Enable hyperparameter tuning using GridSearchCV. Defaults to False."
    )
    parser.add_argument(
        "-mt", "--model_type",
        type=str,
        choices=["xgb", "et"],
        help="Model name to train (xgb, et) (optional, defaults to all)."
    )
    parser.add_argument(
        "-ts", "--ticker_symbol",
        type=str,
        help="""Ticker Symbol to train on. (optional, defaults to all).
        Example: bova11 -> BOVA11.SA | petr4 -> PETR4.SA"""
    )
    parser.add_argument(
        "-sm", "--save_model",
        action="store_false",
        help="Disable saving model to file system. Defaults to True. Run '--save_model' to Disable."
    )
    args = parser.parse_args()

    logger.info("Starting the training pipeline...")
    training_pipeline(args.tune, args.model_type, args.ticker_symbol, args.save_model)
    logger.info("Training Pipeline completed successfully!")

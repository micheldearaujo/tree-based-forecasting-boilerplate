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
    hyperparams_config = yaml.safe_load(f.read())
    all_param_distributions = hyperparams_config['param_spaces']
    all_base_params = hyperparams_config['base_params']

with open("src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())
    model_config = config['model_config']
    data_config = config['data_config']
    PROCESSED_DATA_PATH = data_config['paths']['processed_data_path']
    PROCESSED_DATA_NAME = data_config['table_names']['processed_table_name']
    MODELS_PATH = data_config['paths']['models_path']
    TARGET_COL = model_config['target_col']
    CATEGORY_COL = model_config['category_col']


def train_model(X_train, y_train, model_type, ticker_symbol, tune_params, save_model):
    """Trains a tree-based regression model."""

    
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH+f'/{model_type}', exist_ok=True)
    model_file_path = f"{MODELS_PATH}/{model_type}/{ticker_symbol}.joblib"

    base_params = all_base_params[model_type]

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

    logger.warning(f"Performing hyperparameter tuning for ticker [{ticker_symbol}] using {model_type}...")

    model = xgb.XGBRegressor() if model_type == "XGB" else ExtraTreesRegressor()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    param_distributions = all_param_distributions[model_type]

    grid_search = GridSearchCV(
        model,
        param_grid=param_distributions,
        cv=tscv,
        n_jobs=-1,
        scoring=model_config['scoring_metric'],
        verbose=1
    ).fit(X, y)
    
    best_params = grid_search.best_params_
    logger.warning(f"Best parameters found: {best_params}")
    return best_params


def split_feat_df_Xy(df):
    """Splits the featurized dataframe to train the ML models."""
    X_train=df.drop([TARGET_COL, "DATE"], axis=1)
    y_train=df[TARGET_COL]

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
    logger.debug("Loading the featurized dataset..")
    feature_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_NAME), parse_dates=["DATE"])
    logger.info(f"Last Available Date in Training Dataset: {feature_df['DATE'].max()}")

    # Check the ticker_symbol parameter
    if ticker_symbol:
        ticker_symbol = ticker_symbol.upper() + '.SA'
        feature_df = feature_df[feature_df[CATEGORY_COL] == ticker_symbol]

    # Check the model_type parameter
    available_models = model_config['available_models']
    if model_type is not None and model_type not in available_models:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from: {available_models}")
    
    elif model_type:
        available_models = [model_type]
    
    for ticker_symbol in feature_df[CATEGORY_COL].unique():
        ticker_df_feat = feature_df[feature_df[CATEGORY_COL] == ticker_symbol].drop(CATEGORY_COL, axis=1).copy()
        X_train, y_train = split_feat_df_Xy(ticker_df_feat)

        for model_type in available_models:
            logger.info(f"Training model [{model_type}] for ticker [{ticker_symbol}]...")
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
        choices=["XGB", "ET"],
        help="Model name to train (XGB, ET) (optional, defaults to all)."
    )
    parser.add_argument(
        "-ts", "--ticker_symbol",
        type=str,
        help="""Ticker Symbol to train on. (optional, defaults to all).
        Example: bova11 -> BOVA11.SA | petr4 -> PETR4.SA"""
    )
    parser.add_argument(
        "-dsm", "--dont_save_model",
        action="store_false",
        help="Disable saving model to file system. Defaults to True. Run '--dont_save_model' to Disable."
    )
    args = parser.parse_args()

    logger.info("Starting the training pipeline...")
    training_pipeline(args.tune, args.model_type, args.ticker_symbol, args.dont_save_model)
    logger.info("Training Pipeline completed successfully!")

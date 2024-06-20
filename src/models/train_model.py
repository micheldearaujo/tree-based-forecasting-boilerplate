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
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

with open("./src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)

with open("./src/configuration/hyperparams.yaml", 'r') as f:  
    hyperparams_config = yaml.safe_load(f.read())
    all_param_distributions = hyperparams_config['param_spaces']
    all_base_params = hyperparams_config['base_params']

with open("./src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())
    model_config = config['model_config']
    data_config = config['data_config']
    PROCESSED_DATA_PATH = data_config['paths']['processed_data_path']
    PROCESSED_DATA_NAME = data_config['table_names']['processed_table_name']
    MODELS_PATH = data_config['paths']['models_path']
    TARGET_COL = model_config['target_col']
    CATEGORY_COL = model_config['category_col']


def train_model(X_train, y_train, model_type, ticker_symbol, tune_params, save_model):
    """Trains a tree-based regression model (XGBoost or ExtraTrees).

    This function trains either an XGBoost or ExtraTreesRegressor model for a given ticker symbol.
    It can optionally tune the model hyperparameters using grid search and save the trained model to disk.

    Args:
        X_train (pandas.DataFrame or numpy.ndarray): The training feature data.
        y_train (pandas.Series or numpy.ndarray): The training target values.
        model_type (str): The type of model to train. Choose from 'XGB' (XGBoost) or 'ET' (ExtraTrees).
        ticker_symbol (str): The ticker symbol representing the time series being modeled.
        tune_params (bool, optional): Whether to tune hyperparameters using grid search. Defaults to False.
        save_model (bool, optional): Whether to save the trained model to disk. Defaults to True.

    Returns:
        The trained regression model object (xgb.XGBRegressor or ExtraTreesRegressor).

    Raises:
        ValueError: If an invalid `model_type` is provided.

    Notes:
        - If `tune_params` is True, a grid search will be performed using predefined hyperparameter grids to find the best set of parameters for the given model type.
        - The trained model will be saved in the directory specified by `MODELS_PATH` (which should be defined elsewhere in your code)
            using the format "{model_type}/{ticker_symbol}.joblib".
    """

    
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH+f'/{model_type}', exist_ok=True)
    model_file_path = f"{MODELS_PATH}/{model_type}/{ticker_symbol}.joblib"

    base_params = all_base_params[model_type]

    if tune_params:
        best_params = tune_params_gridsearch(X_train, y_train, model_type, ticker_symbol)
        base_params.update(best_params)

    if model_type == 'XGB':
        model = xgb.XGBRegressor(objective='reg:squarederror', **base_params, eval_metric=["rmse", "logloss"]) \
            .fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=10)
    elif model_type == 'ET':
        model = ExtraTreesRegressor(**base_params).fit(X_train, y_train, verbose=10)
    elif model_type == 'ADA':
        model = AdaBoostRegressor(**base_params).fit(X_train, y_train, verbose=10)
    else:
        raise ValueError("Model type not recognized! Check 'models_list' parameter in project_config.yaml.")
        

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

    if model_type == 'XGB':
        model = xgb.XGBRegressor()
    elif model_type == 'ET':
        model = ExtraTreesRegressor()
    elif model_type == 'ADA':
        model = AdaBoostRegressor()
    else:
        raise ValueError("Model type not recognized! Check 'models_list' parameter in project_config.yaml.")

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

   
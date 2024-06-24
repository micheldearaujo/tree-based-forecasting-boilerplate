# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

import os
import yaml
import argparse
from typing import Any
import logging
import logging.config
import joblib
from dateutil.relativedelta import relativedelta

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    AdaBoostRegressor,
    RandomForestRegressor
)
from sklearn.metrics import root_mean_squared_error

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
    FORECAST_HORIZON = model_config['forecast_horizon']


all_param_distributions['ADA']['estimator'] = [
    eval(model_str) for model_str in all_param_distributions['ADA']['estimator']
]


def split_feat_df_Xy(df):
    """Splits the featurized dataframe to train the ML models."""
    X_train=df.drop([TARGET_COL, "DATE"], axis=1)
    y_train=df[TARGET_COL]

    return X_train, y_train


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str, ticker_symbol: str, load_best_params=True) -> Any:
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
        The trained regression model object.

    Raises:
        ValueError: If an invalid `model_type` is provided.

    Notes:
        - If `tune_params` is True, a grid search will be performed using predefined hyperparameter grids to find the best set of parameters for the given model type.
        - The trained model will be saved in the directory specified by `MODELS_PATH` (which should be defined elsewhere in your code)
            using the format "{model_type}/{ticker_symbol}.joblib".
    """
    base_params = all_base_params[model_type]

    if load_best_params:
        best_params_path = os.path.join(MODELS_PATH, ticker_symbol, f"best_params_{model_type}.joblib") 
        best_params = joblib.load(best_params_path)
        base_params.update(best_params)

        if model_type == 'XGB':
            model = xgb.XGBRegressor(objective='reg:squarederror', **base_params, eval_metric=["rmse", "logloss"]) \
                .fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=20)
        elif model_type == 'ET':
            model = ExtraTreesRegressor(**base_params).fit(X_train, y_train)
        elif model_type == 'ADA':
            model = AdaBoostRegressor(**base_params).fit(X_train, y_train)
        else:
            raise ValueError("Model type not recognized! Check 'models_list' parameter in project_config.yaml.")
        
    else:
        if model_type == 'XGB':
            model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric=["rmse", "logloss"]) \
                .fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=20)
        elif model_type == 'ET':
            model = ExtraTreesRegressor().fit(X_train, y_train)
        elif model_type == 'ADA':
            model = AdaBoostRegressor().fit(X_train, y_train)
        else:
            raise ValueError("Model type not recognized! Check 'models_list' parameter in project_config.yaml.")

    return model


def tune_params_gridsearch(X: pd.DataFrame, y: pd.Series, model_type: str, n_splits=2):
    """
    Performs time series hyperparameter tuning on a model using grid search.
    
    Args:
        X (pd.DataFrame): The input feature data
        y (pd.Series): The target values
        model_type (str): The model to tune. Options: ['XGB', 'ET', 'ADA']
        n_splits (int): Number of folds for cross-validation (default: 3)
    
    Returns:
        best_params (dict): The best hyperparameters found by the grid search
    """

    if model_type == 'XGB':
        model = xgb.XGBRegressor()
    elif model_type == 'ET':
        model = ExtraTreesRegressor()
    elif model_type == 'ADA':
        model = AdaBoostRegressor()
    else:
        raise ValueError("Model type not recognized! Check 'models_list' parameter in project_config.yaml.")

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=FORECAST_HORIZON)
    param_distributions = all_param_distributions[model_type]

    grid_search = GridSearchCV(
        model,
        param_grid=param_distributions,
        cv=tscv,
        n_jobs=6,
        scoring=model_config['scoring_metric'],
        verbose=True,
        return_train_score=True
    ).fit(X, y)
  
    return grid_search

   
def select_and_stage_best_model(models: dict, X_test: pd.DataFrame, y_test: pd.Series, metric='rmse'):
    """
    Evaluates multiple models, selects the best based on a given metric, and stages it to "prod".

    Args:
        models (dict): A dictionary of models with their names as keys (e.g., {'XGB': xgb_model, 'ET': et_model, ...}).
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test target values.
        metric (str, optional): Evaluation metric ('rmse' or 'mae'). Defaults to 'rmse'.
    """
    results = {}

    # Evaluate each model
    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        if metric == 'rmse':
            score = root_mean_squared_error(y_test, y_pred)
        else:
            raise ValueError(f"Invalid metric: {metric}. Choose 'rmse' or 'mae'.")

        results[model_name] = score

    # Select the best model
    logger.info(f"Model Selection Results:\n {results}")
    best_model_name = min(results, key=results.get)  # Get model with lowest error

    logger.info(f"\nBest Model: {best_model_name} with {metric.upper()}: {results[best_model_name]:.4f}")

    return best_model_name

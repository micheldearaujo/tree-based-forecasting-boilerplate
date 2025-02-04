# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import os
import yaml
import datetime as dt
import logging
import logging.config
import argparse
from typing import Any

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)
from pmdarima import auto_arima

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

from src.models.train import train
from src.models.predict import (
    update_lag_features,
    update_target_lag_features,
    update_sma_features,
    update_target_sma_features,
    update_target_ms_features,
    update_bollinger_bands,
    update_rsi,
    update_target_comparison_lag_features,
    update_target_cummax_features
)
from src.visualization.data_viz import visualize_validation_results, extract_learning_curves, plot_tree_feature_importance
from src.configuration.config_data import *
from src.configuration.config_model import *
from src.models.train import split_feat_df_Xy
from src.models.evaluate import update_test_values

from src.configuration.config_feature import *
from src.configuration.config_viz import *
from src.utils.decorators import time_logger

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("git.cmd").setLevel(logging.WARNING)
    logging.getLogger("LightGBM").setLevel(logging.CRITICAL)
    logging.getLogger("lightgbm").setLevel(logging.CRITICAL)
    logger = logging.getLogger(__name__)


PROCESSED_DATA_PATH = data_config['paths']['processed_data_path']
PROCESSED_DATA_NAME = data_config['table_names']['processed_table_name']
OUTPUT_DATA_PATH = data_config['paths']['output_data_path']
OUTPUT_DATA_NAME = data_config['table_names']['output_table_name']
DAILY_PERFORMANCE_DATA_NAME = data_config['table_names']['model_performance_table_name']
CROSS_VAL_DATA_NAME = data_config['table_names']['cross_validation_table_name']
MODELS_PATH = data_config['paths']['models_path']
TARGET_COL = model_config['target_col']
PREDICTED_COL = model_config['predicted_col']
FORECAST_HORIZON = model_config['forecast_horizon']

N_SPLITS = model_config['n_windows']


def get_hopt_search_space(model_type: str) -> dict:
    """
    This function returns a hyperparameter search space for a given machine learning model.

    Args:
         model_type (str): The type of machine learning model for which the hyperparameter search space is required.
                        Supported model types are 'XGBRegressor', 'LGBMRegressor', and 'CatBoostRegressor'.

    Returns:
        dict: A dictionary containing the hyperparameter search space for the specified model type.
            The keys are the hyperparameter names, and the values are lists of possible values for each hyperparameter.

    Raises:
        ValueError: If the specified `model_type` is not supported.
    """
    if model_type == 'XGBRegressor':
        return {
            'n_estimators': hp.choice('n_estimators', [15, 20, 40, 60, 80, 100, 200, 250]),
            'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8, 9, 10]),
            'learning_rate': hp.choice('learning_rate', [0.01, 0.05, 0.1, 0.2, 0.3]),
            'subsample': hp.choice('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
            'colsample_bytree': hp.choice('colsample_bytree', [0.6, 0.7, 0.8, 0.9, 1.0]),
            'reg_lambda': hp.choice('reg_lambda', [0.01, 0.1, 1, 10, 100]),
            'reg_alpha': hp.choice('reg_alpha', [0.01, 0.1, 1, 10, 100]),
        }

    elif model_type == 'LGBMRegressor':
        return {
            'n_estimators': hp.choice('n_estimators', [15, 20, 40, 60, 80, 100, 200, 250]),
            'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8, 9, 10]),
            'learning_rate': hp.choice('learning_rate', [0.01, 0.05, 0.1, 0.2, 0.3]),
            'subsample': hp.choice('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
            'colsample_bytree': hp.choice('colsample_bytree', [0.6, 0.7, 0.8, 0.9, 1.0]),
            'reg_lambda': hp.choice('reg_lambda', [0.01, 0.1, 1, 10, 50, 100]),
            'reg_alpha': hp.choice('reg_alpha', [0.01, 0.1, 1, 10, 50, 100]),
            'verbose': hp.choice('verbose', [-1]),
        }

    elif model_type == 'CatBoostRegressor':
        return {
            'iterations': hp.choice('iterations', [200, 400, 500, 600, 800, 1000, 2000]),
            'depth': hp.choice('depth', [3, 4, 5, 6, 7, 8, 9, 10]),
            'learning_rate': hp.choice('learning_rate', [0.01, 0.05, 0.1, 0.2, 0.3]),
            'subsample': hp.choice('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
            'reg_lambda': hp.choice('reg_lambda', [0.01, 0.1, 1, 10, 50, 100]),
            'verbose': hp.choice('verbose', [2000])
        }

    elif model_type == 'RandomForestRegressor':
        return {
            'n_estimators': hp.choice('n_estimators', [10, 50, 100, 200, 300, 400, 500]),
            'max_depth': hp.choice('max_depth', [None, 5, 10, 15, 20, 25, 30]),
            'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
            'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
            'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': hp.choice('bootstrap', [True, False]),
        }

    elif model_type == 'ExtraTreesRegressor':
        return {
            'n_estimators': hp.choice('n_estimators', [10, 50, 100, 200, 300, 400, 500]),
            'max_depth': hp.choice('max_depth', [None, 5, 10, 15, 20, 25, 30]),
            'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
            'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
            'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': hp.choice('bootstrap', [True, False]),
        }

    elif model_type == 'Ridge':
        return {
            'alpha': hp.choice('alpha', [0.01, 0.1, 1, 10, 100, 1000]),
            'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
        }

    elif model_type == 'SVR':
        return {
            'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'C': hp.choice('C', [0.1, 1, 10, 100, 1000]),
            'gamma': hp.choice('gamma', ['scale', 'auto'] + [0.001, 0.01, 0.1, 1]),
            'epsilon': hp.choice('epsilon', [0.01, 0.1, 0.5, 1, 2]),
        }

    else:
        raise ValueError(f"Invalid model name: {model_type}")


def objective(base_model, params: dict, X: pd.DataFrame, y: pd.Series, forecast_horizon: int) -> dict:
    """
    Evaluates a time series model using a rolling forecast approach.

    This function performs a rolling forecast for a given time series model,
    dynamically updates features, and calculates the mean squared error. It's
    designed for model evaluation and hyperparameter tuning in time series
    forecasting tasks.

    Args:
        base_model (object): The machine learning model to be evaluated 
                                (e.g., sklearn estimator).
        params (dict): Hyperparameters for the base_model.
        X (pd.DataFrame): Feature set including a 'date' column and other predictors.
        y (pd.Series): Target variable.
        forecast_horizon (int): Number of future time steps to forecast.

    Returns:
        dict: A dictionary containing:
            - 'loss' (float): Mean squared error of the forecast.
            - 'status' (str): Status of the optimization.
    """

    # Create empty list for storing each prediction
    predictions = []
    actuals = []
    dates = []
    X_testing_df = pd.DataFrame()

    X_train = X.iloc[:-forecast_horizon, :]
    y_train = y.iloc[:-forecast_horizon]
    final_y = y_train.copy()

    base_model.fit(X_train.drop(columns=["date"]), y_train)

    for day in range(forecast_horizon, 0, -1):
        X_test, y_test = update_test_values(X, y, day)
        logger.debug(f"Testing Date: {X_test['date'].min().date()}")

        if len(predictions) != 0:

            X_test = update_target_lag_features(X_test, -1, list(final_y.values), X_test.columns)
            X_test = update_target_sma_features(X_test, -1, list(final_y.values), X_test.columns)
            X_test = update_target_ms_features(X_test, -1, list(final_y.values), X_test.columns)
            X_test = update_bollinger_bands(X_test, -1, list(final_y.values), X_test.columns)
            X_test = update_target_comparison_lag_features(X_test, -1, list(final_y.values), X_test.columns)
  
        prediction = base_model.predict(X_test.drop("date", axis=1))

        # store the results
        predictions.append(prediction[0])
        actuals.append(y_test.values[0])
        dates.append(X_test["date"].max())

        final_y = pd.concat([final_y, pd.Series(prediction[0])], axis=0)
        final_y = final_y.reset_index(drop=True)
        X_testing_df = pd.concat([X_testing_df, X_test], axis=0)

    pred_df = pd.DataFrame(list(zip(dates, actuals, predictions)), columns=["date", "ACTUAL", PREDICTED_COL])
    loss = mean_squared_error(pred_df["ACTUAL"], pred_df[PREDICTED_COL])
    
    return {'loss': loss, 'status': STATUS_OK}


# Module 4: Hyperparameter optimization
@time_logger
def tune_model(
    model_type:     str,
    X:              pd.DataFrame,
    y:              pd.Series,
    forecast_horizon: int,
    max_evals:      int=20
    ) ->            dict:
    """
    This function performs hyperparameter optimization for a given machine learning model using the hyperopt library.
    It uses the objective function to evaluate the performance of different hyperparameter combinations.

    Args:
        - model_type (str): The type of machine learning model to be tuned. Supported types are 'XGBRegressor', 'LGBMRegressor', and 'CatBoostRegressor'.
        - X (pd.DataFrame): The feature DataFrame containing all historical data.
        - y (pd.Series): The target Series containing all historical target values.
        - forecast_horizon (int): The number of days ahead for which predictions are required.

    Returns:
        - best_params (dict): A dictionary containing the best hyperparameters found during the optimization process.
        - trials (Trials): An object containing information about the trials performed during the optimization process.
    """
    search_space = get_hopt_search_space(model_type)
    trials = Trials()

    base_model = get_models()[model_type]

    results = fmin(
        fn=lambda params: objective(base_model, params, X, y, forecast_horizon),
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )

    best_params = space_eval(search_space, results)

    return best_params, trials


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

    model = model_mapping.get(model_type)()

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=FORECAST_HORIZON)
    param_distributions = tuning_param_spaces[model_type]

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


gridsearch_param_spaces = {
    'XGBRegressor': {
        'n_estimators': [20, 35, 40, 60, 80, 100],
        'max_depth': [2, 3, 4, 6, 13],
        'learning_rate': [0.1, 0.05, 0.01],
        # 'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.3, 0.5, 0.8, 1.0],
        'reg_lambda': [0.01, 0.3, 0.9, 1.5, 3, 5, 10, 20, 30],
        'seed': [42]
    },
    'LGBMRegressor': {
        'boosting_type': ['gbdt', 'dart'],
        'n_estimators': [20, 30, 60, 100, 200],
        'max_depth': [3, 4, 6, 9, 12, 15],
        'learning_rate': [0.1, 0.05, 0.01],
        # 'num_leaves': [31, 50, 100],
        'colsample_bytree': [0.3, 0.5, 0.8, 1.0],
        'reg_lambda': [0.01, 0.3, 0.9, 1.5, 3, 5, 10, 20 ,30],
        # 'min_child_samples': [10, 20, 30],
        'seed': [42],
        'verbose': [-1]
    },
    'CatBoostRegressor': {
        'iterations': [40, 100, 500, 1000, 1500],
        'depth': [2, 3, 4, 5, 13],
        'learning_rate': [0.1, 0.05, 0.01],
        'l2_leaf_reg': [1, 2, 3, 7, 10, 15, 20, 30],
        'random_seed': [42]
    },
    'RandomForestRegressor': {
        'n_estimators': [50, 100, 500],
        'max_depth': [None, 2, 3, 4, 5, 7, 13],
        'max_features': ['auto', 0.4, 0.6, 0.8, 1.0],
        # 'min_samples_split': [2, 5, 10],
        # 'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'random_state': [42]
    }
}
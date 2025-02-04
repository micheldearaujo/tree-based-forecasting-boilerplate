import sys
import os

sys.path.insert(0, ".")
import warnings
import numpy as np
import pandas as pd
from typing import Any
import joblib
import json

import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from pmdarima import auto_arima

from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
import shap
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.configuration.config_data import *
from src.configuration.config_model import *
from src.configuration.config_feature import *
from src.configuration.config_viz import *
import src.utils.useful_functions as uf
from src.features.feat_eng import *
from src.utils.decorators import time_logger

warnings.filterwarnings("ignore")
plt.switch_backend('agg')

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
MODELS_PATH         = data_config['paths']['models_path']
TARGET_COL          = model_config['target_col']
FORECAST_HORIZON    = model_config['forecast_horizon']
MODEL_NAME          = model_config["model_name"]
EXPERIMENTS_PATH    = model_config['mlflow_experiment_path']
MLRUNS_PATH         = model_config['mlflow_runs_path']
USE_TUNED_PARMS     = model_config["use_tuned_params"]
models_list         = list(get_models().keys())

@time_logger
def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    load_best_params: bool=True
) -> Any:
    """
    Trains a regression model.

    This function trains either an Tree-based model or classical Regression model..
    It can optionally tune the model hyperparameters using grid search and save the trained model to disk.

    Args:
        X_train (pandas.DataFrame or numpy.ndarray): The training feature data.
        y_train (pandas.Series or numpy.ndarray): The training target values.
        model_type (str): The type of model to train. Choose from get_models funcion in config_model.
        ticker_symbol (str): The ticker symbol representing the time series being modeled.
        load_best_params (bool): If True, loads saved best model parameters; otherwise, uses default parameters. Defaults to True.

    Returns:
        The trained regression model object.
    """

    try:
        model_instance = get_models()[model_type]
    except:
        raise ValueError(f"Invalid model name: {model_type}")

    if load_best_params:
        logger.info("Loading the best parameters from the ./models folder...")
        best_params_path = f"{MODELS_PATH}/best_params_{model_type}_{model_flavor}.json"

        if os.path.exists(best_params_path):
            with open(best_params_path) as f:
                best_params = json.load(f)

            model_instance.set_params(**best_params)
        else:
            logger.warning(f"Best parameters file not found. Continuing with default...")

    y_train_arima = y_train.copy()
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42)
    eval_set = [(X_train, y_train), (X_val, y_val)]

    if model_type == 'ARIMA':
        model = auto_arima(
            y = y_train_arima,
            max_p = 2,
            max_d = 1,
            max_q = 2,
            seasonal=False, trace=False, error_action='ignore', suppress_warnings=True
        )

    elif model_type in ["XGBRegressor", "CatBoostRegressor", "LGBMRegressor"]:
        try:
            model = model_instance.fit(X_train, y_train, eval_set=eval_set, verbose=10)
        except:
            model = model_instance.fit(X_train, y_train, eval_set=eval_set)
    else:
        model = model_instance.fit(X_train, y_train)

    return model

def calculate_shap_values_to_df(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate SHAP values for a tree-based model and store them in a DataFrame.

    Args:
        model: Trained tree-based model (e.g., XGBoost, LightGBM).
        X (pd.DataFrame): Testing dataset to calculate SHAP on.

    Returns:
        pd.DataFrame: DataFrame containing SHAP values for the test set.
    """
    model_type = type(model).__name__
    if model_type in ["ARIMA", "VotingRegressor"]:
        shap_values = np.zeros(X.shape)
        shap_values_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
        return shap_values, shap_values_df

    try:
        if model_type in ["Ridge"]:
            explainer = shap.LinearExplainer(model, X)
        elif model_type in ["SVR", "LinearSVR"]:
            explainer = shap.KernelExplainer(model.predict, X)
        else:
            explainer = shap.TreeExplainer(model)

    except Exception as e:
        raise AttributeError(f"Failed to initialize SHAP explainer for model type {model_type}. Error: {str(e)}")

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X)
    # Convert SHAP values into a DataFrame
    shap_values_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)

    return shap_values, shap_values_df


def split_feat_df_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Splits the featurized dataframe to train the ML models."""
    X_train=df.drop([TARGET_COL], axis=1)
    y_train=df[TARGET_COL]

    return X_train, y_train
    

def configure_mlflow_experiment(
    experiments_path: str = EXPERIMENTS_PATH,
    mlruns_path: str = MLRUNS_PATH,
    experiment_id: str = None
) -> None:
    """
    Configures the MLflow experiment and tracking URI.

    This function sets up the MLflow tracking URI to the specified path
    and sets the experiment to the provided name.
    It also enables automatic logging of parameters, metrics, and artifacts
    for all subsequent runs within the experiment.

    Args:
        experiments_path (str): The path to the directory where the MLflow experiment will be stored.
            Defaults to the value specified in the EXPERIMENTS_PATH constant.
        mlruns_path (str): The path to the directory where the MLflow tracking data will be stored.
            Defaults to the value specified in the MLRUNS_PATH constant.

    Returns:
        None
    """
    mlflow.set_tracking_uri(mlruns_path)
    if experiment_id is None:
        mlflow.set_experiment(experiment_name=experiments_path)
    else:
        mlflow.set_experiment(experiment_id=experiment_id)
    mlflow.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)


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
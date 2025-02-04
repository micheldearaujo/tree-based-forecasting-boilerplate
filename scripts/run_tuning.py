# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import pandas as pd
import pandas_gbq

import os
import warnings
import yaml
import setuptools
import datetime as dt
import time
import logging
import logging.config
import argparse
from dateutil.relativedelta import relativedelta
from typing import Any

import pandas as pd
import numpy as np
import pmdarima as pm

import mlflow.sklearn
from mlflow.models import infer_signature
import shap

from src.models.evaluate import *
from src.models.evaluate import stepwise_prediction
from src.visualization.data_viz import *
from src.models.train import *
from src.models.predict import forecast_external_features
import src.features.feat_eng as fe
from src.data.data_loader import load_and_preprocess_model_dataset
from src.models.tune_params import tune_model
from src.utils.decorators import time_logger

warnings.filterwarnings("ignore")


with open("src/configuration/logging_config.yaml", 'r') as f:
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("git.cmd").setLevel(logging.WARNING)
    logging.getLogger("LightGBM").setLevel(logging.CRITICAL)
    logging.getLogger("lightgbm").setLevel(logging.CRITICAL)
    logger = logging.getLogger(__name__)



def run_tuning_pipeline(dataframe, max_evals, models_list):
    """
    Runs a hyperparameter tuning pipeline for a list of models on the provided dataset.

    This function performs the following steps:
    1. Logs the last available date in the dataset.
    2. Splits the dataset into features (X) and target (y) for tuning.
    3. Iterates over the list of models, tuning each one using the specified number of evaluations.
    4. Saves the best hyperparameters for each model to JSON and joblib files.

    Args:
        dataframe (pd.DataFrame): The dataset containing features and target for tuning.
        max_evals (int): The maximum number of hyperparameter evaluations to perform for each model.
        models_list (list): A list of model types (e.g., ['xgboost', 'random_forest']) to tune.

    Returns:
        None: The function does not return any value but saves the best hyperparameters to disk.
    """

    logger.info(f"Last Available Date for Tuning: {dataframe['date'].max()}")

    X_tuning, y_tuning = split_feat_df_Xy(tuning_df)

    for model_type in models_list:

        best_params, trials     = tune_model(
            model_type          = model_type,
            X                   = X_tuning,
            y                   = y_tuning,
            forecast_horizon    = FORECAST_HORIZON,
            max_evals           = max_evals
        )

        logger.info(f"Best parameters for {model_type}: {best_params}")
        os.makedirs(MODELS_PATH, exist_ok=True)
        
        with open(f"./models/best_params_{model_type}_{model_flavor}.json", "w") as outfile:
            json.dump(best_params, outfile)

        joblib.dump(best_params, f"./models/best_params_{model_type}_{model_flavor}.joblib")


if __name__ == "__main__":

    TARGET_COL          = model_config["target_col"]
    PREDICTED_COL       = model_config["predicted_col"]
    FORECAST_HORIZON    = model_config["forecast_horizon"]
    N_SPLITS            = model_config["n_windows"]
    MODEL_NAME          = model_config["model_name"]
    USE_TUNED_PARMS     = model_config["use_tuned_params"]
    TUNING_HOLDOUT_DATE = model_config["tuning_holdout_date"]
    models_list         = list(get_models().keys())

    feature_df = load_and_preprocess_model_dataset("featurized_df")

    tuning_df = feature_df[feature_df["date"] <= TUNING_HOLDOUT_DATE].copy()
    

    run_tuning_pipeline(
        dataframe = tuning_df,
        max_evals = 4000,
        models_list = models_list
    )
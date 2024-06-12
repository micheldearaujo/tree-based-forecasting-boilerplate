# -*- coding: utf-8 -*-
# ------------------------
# - Utilities script -
# ------------------------

import sys
import os

sys.path.insert(0,'.')

import yaml
import logging
import logging.config

import pandas as pd
import numpy as np
import xgboost as xgb

from src.config import *


with open("src/configuration/hyperparams.yaml", 'r') as f:  

    model_config = yaml.safe_load(f.read())

with open("src/configuration/logging_config.yaml", 'r') as f:  

    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)

# Organizing all parameter distributions into one dictionary
# param_space_dict = { 
#     'ExtraTreesRegressor': model_config['PARAM_SPACES']['et'],
#     'XGBRegressor': model_config['PARAM_SPACES']['xgb'],
#     'LightGBM': model_config['PARAM_SPACES']['lgb']
# }

def write_dataset_to_file(df: pd.DataFrame, dir_path: str, file_name: str) -> None:
    """Saves any DataFrame to a CSV file in the specified directory."""

    os.makedirs(dir_path, exist_ok=True)
    df.to_csv(os.path.join(dir_path, f'{file_name}.csv'), index=False)


def weekend_adj_forecast_horizon(original_forecast_horizon, weekend_days_per_week):

    adjusted_forecast_horizon = original_forecast_horizon - int(original_forecast_horizon * weekend_days_per_week / 7)

    return adjusted_forecast_horizon

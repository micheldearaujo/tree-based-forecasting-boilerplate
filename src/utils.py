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

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

with open("src/configuration/hyperparams.yaml", 'r') as f:  
    hyperparams_config = yaml.safe_load(f.read())

with open("src/configuration/logging_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)

# Organizing all parameter distributions into one dictionary
# param_space_dict = { 
#     'ExtraTreesRegressor': hyperparams_config['PARAM_SPACES']['et'],
#     'XGBRegressor': hyperparams_config['PARAM_SPACES']['xgb'],
#     'LightGBM': hyperparams_config['PARAM_SPACES']['lgb']
# }


def write_dataset_to_file(df: pd.DataFrame, dir_path: str, file_name: str) -> None:
    """Saves any DataFrame to a CSV file in the specified directory."""

    os.makedirs(dir_path, exist_ok=True)
    df.to_csv(os.path.join(dir_path, file_name), index=False)


def weekend_adj_forecast_horizon(original_forecast_horizon, weekend_days_per_week):

    adjusted_forecast_horizon = original_forecast_horizon - int(original_forecast_horizon * weekend_days_per_week / 7)

    return adjusted_forecast_horizon

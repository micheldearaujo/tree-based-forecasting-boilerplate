import sys
sys.path.insert(0,'.')

import numpy as np
from src.configuration.config_model import model_config

import logging
import logging.config
import yaml

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("git.cmd").setLevel(logging.WARNING)
    logging.getLogger("LightGBM").setLevel(logging.CRITICAL)
    logging.getLogger("lightgbm").setLevel(logging.CRITICAL)
    logger = logging.getLogger(__name__)


columns_names_mapping = {
    'china_port_inventories': 'inventories',
    'woodchips_imports': 'imports',
    'resale_bhkp_usd': 'second_market_price',
    'sci_market_price_rmbt': 'final_product_price',
    'pix_china_bhkp_net_usdt': 'index_price'
}

################## Variables ##################
variables_list = [
  'inventories',
  'imports',
  'europulp',
  'second_market_price',
  'final_product_price',
]

################## Feature Engineering ##################
global model_flavor
model_flavor = 'endogenous'

if model_flavor == 'mix':
  lags_target_list = [1, 2]               # <- List of lags of the target variable to use as feature
  sma_target_values = [3, 12]             # <- List of moving average windows of the target variable to use as feature
  moving_sum_target_values = []           # <- List of moving sum windows of the target variable to use as feature
  bollinger_bands_values = [8, 20]        # <- List of Bollinger Bands windows of the target variable to use as feature
  lags_comparison_target_dict = {
    model_config['target_col']: [[4, 5]]  # <- List of lags to calculate division. Lag[0] / Lag[1]
  }

  ms_comparison_target_dict = {
    # model_config['target_col']: [[6, 12]]  # <- List of moving sum windows to calculate division. moving_sum[0] / moving_sum[1]
  }

  sma_comparison_target_dict = {
    # model_config['target_col']: [[6, 12]]   # <- List of moving average windows to calculate division. moving_sum[0] / moving_sum[1]
  }

  rsi_windows_list = []                      # <- List of RSI windows

  spread_features_dict = {                 # <- List of features to calculate spread (subtraction). feature_1 - feature_2
    "index_vs_market_price": [model_config['target_col'], "second_market_price"]
  }
  
  lags_exog_dict = {                       # <- Dictionary of lag values to apply in each of the external variables
    "imports": [6],
    "inventories": [16, 20],               # <- You can use more than one lag for each variable
    "europulp": [6, 14],
    "second_market_price": [3],
    "final_product_price": [2],
  }

  sma_exog_dict = {
    # "second_market_price": [12],             # <- Dictionary of moving average windows of the external variables to use as feature
  }
  sma_comparison_exog_dict = {
    # "second_market_price": [[6, 12]]
  }                                           # <- Dictionary of moving average windows to calculate division of external variables. sma_var[0] / sma_var[1]
  lags_comparison_exog_dict = {
    # "second_market_price": [[6, 12]]
  }                                           # <- Dictionary of lags to calculate division of external variables. Lag[0] / Lag[1]

elif model_flavor == 'endogenous':

  lags_target_list = [1, 2]
  sma_target_values = [3, 12]
  moving_sum_target_values = [3]

  bollinger_bands_values = [8, 20]
  rsi_windows_list = []
  lags_comparison_target_dict = {
    model_config['target_col']: [[4, 5]]
  }
  spread_features_dict = {                 # <- List of features to calculate spread (subtraction). feature_1 - feature_2
    # "index_vs_market_price": [model_config['target_col'], "second_market_price"]
  }
  ms_comparison_target_dict = {}
  sma_comparison_target_dict = {}

  moody_lag_values = []
  lags_exog_dict = {}
  sma_exog_dict = {}
  sma_comparison_exog_dict = {}
  lags_comparison_exog_dict = {}

else:
  raise Exception(f"Unknown Model Class: {model_flavor}")

logger.debug(f"Model Flavor selected: {model_flavor}")

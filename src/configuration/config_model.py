import sys
sys.path.insert(0,'.')

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.svm import SVR, LinearSVR
from pmdarima import auto_arima


global model_config
model_config = {
    'target_col': 'index_price',
    'predicted_col': 'FORECAST',
    'model_name': 'fcst',
    'forecast_horizon': 6,
    'cross_validation_step_size': 6,
    'validation_metric': 'MAPE',
    'optimization_metric': 'MSE',
    'scoring_metric': 'neg_mean_squared_error',
    'model_flavors': ['mix', 'endogenous'], # Different set of features
    'use_tuned_params': False,
    'tuning_holdout_date': '2024-09-13', # 04 and 18. 18 is 1 window of 6 weeks, 04 has 8 weeks
    'mlflow_experiment_path': 'experiments',
    'mlflow_experiment_path_production': 'develop',
    'mlflow_runs_path': './mlruns',
    'model_selection_engine': 'best', # 'best', 'all' for using all models, 'ensemble' for ensemble.
    'model_registry_engine': 'default' # "default" for .picke models, "mlflow" for mlflow model registry

}

################## Model Parameters ##################

median_q = 0.5
lower_q = 0.1
higher_q = 0.9

def get_models():
    """
    This function returns a dictionary of
    machine learning models with their respective parameters.

    Args:
        None

    Returns:
        dict: A dictionary where the keys are the names of the machine learning models and the values are the initialized instances of the models.
    """
    models = {
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        # "XGBRegressor": XGBRegressor(random_state=42),
        # "LGBMRegressor": LGBMRegressor(verbose=-1, random_state=42),
        # "CatBoostRegressor": CatBoostRegressor(use_best_model=False, verbose=200),
        # "ARIMA": "ARIMA",
        # "ExtraTreeRegressor": ExtraTreeRegressor(random_state=42),
        "Ridge": Ridge(random_state=42),
        # "SVR": LinearSVR()
    }

    ensemble = VotingRegressor(
        estimators = [(model_name, model_instance) for model_name, model_instance in models.items() if model_name != 'ARIMA'],
    )

    models['VotingRegressor'] = ensemble

    return models

def get_quantile_models(): 

    models = {
        'XGBoostRegressorQ': {'objective':'reg:quantileerror', 'quantile_alpha':[lower_q, median_q, higher_q],'random_state': 123},
        # 'CatBoostRegressorQM': {'loss_function': f'Quantile:alpha={median_q}', 'verbose': 2000},
        # 'LGBMRegressorQU': {'objective': 'quantile', 'alpha': higher_q, 'boosting_type': 'gbdt','metric': 'quantile'}
    }

    return models
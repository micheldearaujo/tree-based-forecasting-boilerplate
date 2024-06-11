# -*- coding: utf-8 -*-
# ------------------------
# - Utilities script -
# ------------------------


import asyncio
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import date
import yfinance as yfin
import datetime as dt
import sys
import os
import logging
from joblib import load, dump
from scipy.stats import uniform, randint

# Time Series Libraries
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima.utils import ndiffs 

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
# from hyperopt import fmin, tpe, Trials, hp, SparkTrials, space_eval, STATUS_OK, rand, Trials

# front-end
import streamlit as st
import altair as alt
import plotly.express as px

plt.style.use("fivethirtyeight")

# Define dates to start and end
initial_stock_date = dt.datetime.now().date() - dt.timedelta(days=5*365)
final_stock_date = dt.datetime.now().date()

model_config = {
    "TEST_SIZE": 0.2,
    "TARGET_NAME": "Close",
    "VALIDATION_METRIC": "MAPE",
    "OPTIMIZATION_METRIC": "MSE",
    "FORECAST_HORIZON": 7,
    "REGISTER_MODEL_NAME_VAL": "Stock_Predictor_Validation",
    "REGISTER_MODEL_NAME_INF": "Stock_Predictor_Inference",
    "MODEL_NAME": "xgboost_model",
}

features_list = ["day_of_month", "month", "day_of_week", "week_of_month", "quarter", "CLOSE_MA_3", "Close_lag_1", "Close_lag_2"]

# Define a ação para procurar
PERIOD = '3600d'
INTERVAL = '1d'
STOCK_NAME = 'BOVA11.SA'
stocks_list = [
    "BOVA11.SA", "BCFF11.SA",# "MXRF11.SA", "HGLG11.SA", "ITSA4.SA", 
    "TAEE4.SA", #"RAIZ4.SA",
    "EGIE3.SA", "BBSE3.SA", "CSMG3.SA", "PETR4.SA",
    "BRSR6.SA"]


# paths
ROOT_DATA_PATH = "./data"
RAW_DATA_PATH = os.path.join(ROOT_DATA_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(ROOT_DATA_PATH, "processed")
OUTPUT_DATA_PATH = os.path.join(ROOT_DATA_PATH, "output")

MODELS_PATH = "./models"

xgb_base_params = {
    'learning_rate': 0.01,
    'max_depth': 11,
    'n_estimators': 40,
    'reg_lambda': 10,
    'scale_pos_weight': 10,
    'seed': 42,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'gamma': 0.01
}

# xgb_param_space_hpt = {
#     'max_depth': hp.choice('max_depth', [1, 2, 4, 9, 11, 30]),
#     'learning_rate': hp.choice('learning_rate', [0.01, 0.03, 0.05, 0.08 ,0.1, 0.5, 1.0]),
#     'gamma': hp.choice('gamma', [0.005, 0.01, 0.08, 0.1, 1.0]),
#     'reg_lambda': hp.choice('reg_lambda', [1, 10, 30, 40, 50, 60]),
#     'n_estimators': hp.choice('n_estimators', [40, 150, 180, 200, 230, 250, 300]),
#     'scale_pos_weight': hp.choice('scale_pos_weight', [2, 3, 4, 10, 15, 17, 20, 25, 30]),
#     'colsample_bytree': hp.choice('colsample_bytree', [0.8, 1.0]),
#     'min_child_weight': hp.choice('min_child_weight', [1, 2, 4, 7, 8, 10]),
#     'subsample': hp.choice('subsample', [0.8, 1.0]),
#     'reg_alpha': hp.choice('reg_alpha', [0.01, 0.1, 0.25, 0.5, 1.0]),

# }

et_param_space = {
    'n_estimators': [100, 200, 400],
    'max_features': ['1.0', 'sqrt', 'log2'],
    'max_depth': [None, 3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

xgb_param_space = {
    'min_child_weight': [None, 1, 3, 5, 7],
    "n_estimators": [None, 40, 100, 300],
    "max_depth": [None, 3, 5, 7, 9],
    "learning_rate": [None, 0.2, 0.3, 0.1, 0.01],
    "subsample": [None, 0.8, 1.0],
    "colsample_bytree": [None, 1.0],
    "gamma": [None, 0.1, 0.25, 0.5, 1.0],
    "reg_alpha": [None, 0, 0.25, 0.5, 1.0],
    "reg_lambda": [None, 0, 0.25, 0.5, 1.0],
    "seed": [42]
}

lgb_param_space = {
    "min_child_weight": [None, 1, 3, 5, 7],
    "n_estimators": [None, 40, 100, 300],
    "max_depth": [None, 3, 5, 7, 9],
    "learning_rate": [None, 0.2, 0.3, 0.1, 0.01],
    "subsample": [None, 0.8, 1.0],
    "colsample_bytree": [None, 1.0],
    "gamma": [None, 0.1, 0.25, 0.5, 1.0],
    "reg_alpha": [None, 0, 0.25, 0.5, 1.0],
    "reg_lambda": [None, 0, 0.25, 0.5, 1.0],
    "seed": [42]
}

# Organizing all parameter distributions into one dictionary
param_space_dict = {
    'ExtraTreesRegressor': et_param_space,
    'XGBRegressor': xgb_param_space,
    'LightGBM': lgb_param_space
}
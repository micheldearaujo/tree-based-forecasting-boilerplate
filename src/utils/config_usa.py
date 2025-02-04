import os

# !pip install --upgrade pip
# !python -m pip install git+https://github.com/rodrigodmotta/eiapy.git
# !pip install fredapi
# !pip install stats-can

os.environ['EIA_KEY'] = '81df175923880a66568e31bb53783b3d'
#https://www.eia.gov/opendata/qb.php
from eiapy import Series

# import pyspark
import dateutil
# import pyspark.sql.functions as F
import datetime as dt
# import pyspark.sql.types as Types
import numpy as np

import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
from scipy import stats
import math
from scipy.stats import normaltest

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.seasonal import STL


# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
import pmdarima
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf  # Plot de Autocorrelação - Moving Averages
from statsmodels.graphics.tsaplots import plot_pacf  # Plot de Autocorrelação - Auto Regressive
from pmdarima.arima.utils import ndiffs  # Testes para saber o número de diferenciações
from statsmodels.tools.eval_measures import rmse, aic
from pmdarima.arima.utils import ndiffs 
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import prophet
from prophet.utilities import regressor_coefficients


# Classical Machine Learning Libraries
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

# Configuration
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import statsmodels.tsa.holtwinters as ets
from hyperopt import fmin, tpe, Trials, hp, SparkTrials, space_eval, STATUS_OK, rand, Trials
import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature
#import shap

plt.style.use('fivethirtyeight')

FORECAST_HORIZON = 12
PREDICTED_LABEL = 'predicted'
Y_LABEL = 'SALES'
MAX_DATE_FILTER = pd.to_datetime('2023-01-01')

# Define the FORECAST HORIZON - It varies from month to month
training_date = MAX_DATE_FILTER
next_year = training_date + dateutil.relativedelta.relativedelta(years=1)
next_year = (next_year +  dateutil.relativedelta.relativedelta(months = 12 - next_year.month)).date()
FORECAST_HORIZON_FORECAST = (next_year.year - training_date.year) * 12 + (next_year.month - training_date.month) + 1

external_variables_us_list = [
    "retail_diesel_price_cents",
    "retail_gas_price_cents",
    "co2_mill_tons",
    "unemployment_percent",
    "housing_starts_mill",
    "real_gdp_bill_chained",
    "airline_ticket_price_index",
    "steel_production_mill_short_tons",
    "vehicle_miles_traveled_mill_miles_day",
    "consumer_price_index",
    "US_WEEKLY_Gasoline",
    "cpi_used_cars",
    "ppi_industry",
    "ppi_truck",
    "ppi_purchasing_power",
    "confidence_level",
    "durable_goods",
    "personal_saving_rate",
    "consumer_loans",
    "unemployment_rate",
    "cpi_new_vehicles",
    "ppi_tire",
    "inflation_rate",
    "real_gross_domestic_product",
    "light_weight_vehicle_sales",
]

# COMMAND ----------

xgboost_hyperparameter_config = {
    'max_depth': hp.choice('max_depth', [4, 9, 11, 30]),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.08 ,0.1, 0.5, 1.0]),
    'gamma': hp.choice('gamma', [0.01, 0.08, 0.1, 1.0]),
    'reg_lambda': hp.choice('reg_lambda', [1, 10, 30, 100]),
    'n_estimators': hp.choice('n_estimators', [40, 200, 300, 500, 1000]),
    'scale_pos_weight': hp.choice('scale_pos_weight', [1, 2, 3, 4, 10, 15]),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
}

# COMMAND ----------

xgboost_model_config = {
    'LEARNING_RATE': 0.01,
    'MAX_DEPTH': 100,
    'MIN_DATA': 100,
    'N_ESTIMATORS': 1000,
    'REG_LAMBDA': 100,
    'SCALE_POS_WEIGHT': 10,
    'SEED': 42,
    'SUBSAMPLE': 0.9,
    'COLSAMPLE_BYTREE': 0.9,
    'NUM_BOOST_ROUNDS': 200,
    'GAMMA': 0.01
}

# COMMAND ----------

xgboost_fixed_model_config = {
    'SEED': 42,
    'SUBSAMPLE': 0.95
}

# COMMAND ----------

model_config = {
    'TARGET_VARIABLE': 'SALES',
    'COMPARISON_METRIC': 'MAPE',
    'TEST_SIZE': 0.2,
    'MAX_EVALS': 10,
    'REGISTER_MODEL_NAME': 'marketForecastModel',
    'PARALELISM': 1,
    'MAX_EVALS': 200,
}

xgboost_fixed_model_config = {
    'SEED': 42,
    'SUBSAMPLE': 0.95
}

xgboost_hyperparameter_config = {
    'max_depth': hp.choice('max_depth', [4, 9, 11, 30]),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.08 ,0.1, 0.5, 1.0]),
    'gamma': hp.choice('gamma', [0.01, 0.08, 0.1, 1.0]),
    'reg_lambda': hp.choice('reg_lambda', [1, 10, 30, 100]),
    'n_estimators': hp.choice('n_estimators', [40, 200, 300, 500, 1000]),
    'scale_pos_weight': hp.choice('scale_pos_weight', [1, 2, 3, 4, 10, 15]),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
}



# COMMAND ----------

# Dicionary to map features for each series
country_pool_features_mapping_prophet = {
  'UNITED STATES-Pool': {

  },
  'UNITED STATES-Non_Pool': {
    #"co2_mill_tons" : [12],
    "housing_starts_mill": [1],
    
  },
  'CANADA-Pool': {
    #"new_car_sales": [1],
    
  },
  'UNITED STATES-TOTAL': {
    #"vehicle_miles_traveled_mill_miles_day": [1],
    #"housing_starts_mill": [2],
  }
}

# COMMAND ----------

country_pool_features_mapping = {
  'UNITED STATES-Pool': {
    'real_gross_domestic_product': [0],
    'vehicle_miles_traveled_mill_miles_day': [0],
    'light_weight_vehicle_sales': [0]
  },

  'UNITED STATES-Non_Pool': {
    'real_gross_domestic_product': [0],
    'vehicle_miles_traveled_mill_miles_day': [0],
    'light_weight_vehicle_sales': [0]
    
  },
  
  'UNITED STATES-TOTAL': {
    'real_gross_domestic_product': [0],
    'vehicle_miles_traveled_mill_miles_day': [0],
    'light_weight_vehicle_sales': [0]
  }
}

# COMMAND ----------

country_pool_features_mapping_rim_ize = {
  'Any': {}
}

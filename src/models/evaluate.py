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
from src.visualization.data_viz import plot_crossval_results, extract_learning_curves, plot_tree_feature_importance
from src.configuration.config_data import *
from src.configuration.config_model import *
from src.configuration.config_feature import *
from src.configuration.config_viz import *

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("git.cmd").setLevel(logging.WARNING)
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


def CustomTimeSeriesSplit(data: pd.DataFrame, test_start_date: str, test_size: int = 6, step_size: int = 6):
    """
    Perform custom time series split with walking window and overlap on the test set,
    returning indices like TimeSeriesSplit.
    Note: DataFrame's Index must be DatetimeIndex.
    
    Args:
        data: pd.DataFrame with a datetime index.
        test_start_date: The start date for the first test set (datetime or string).
        test_size: The number of periods to use for the test set in each fold (default is 6).
        step_size: The number of periods to walk forward the training set in each iteration (default is 6).
    
    Returns:
        splits: A list of tuples where each tuple contains the indices for the train and test sets.
    """
    
    # Ensure the 'data' has a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DatetimeIndex.")
    
    # Ensure the test_start_date is in datetime format
    test_start_date = pd.to_datetime(test_start_date)
    
    # Find the closest date to test_start_date if exact match is not found
    if test_start_date not in data.index:
        raise ValueError("Closes Date search is not implemented Yet! Choose an exact date on your timeframe!!!")
        closest_date = data.index[(data.index - test_start_date).abs().argmin()]
        print(f"Exact start date not found. Using closest available date: {closest_date}")
        test_start_idx = data.index.get_loc(closest_date)  # Integer index
    else:
        test_start_idx = data.index.get_loc(test_start_date)  # Integer index
    
    # Convert the test size and overlap to Timedelta
    overlap = test_size - step_size
    test_size_timedelta = pd.Timedelta(weeks=test_size)
    overlap_timedelta = pd.Timedelta(weeks=overlap)
    step_size_timedelta = pd.Timedelta(weeks=step_size)
    
    # Initialize list to store the splits
    splits = []
    
    # Calculate the initial indices for the first split (test set will be fixed size of 12 weeks)
    test_start = data.index[test_start_idx]  # Get the start date of the first test set
    test_end = test_start + test_size_timedelta  # The test period will be 12 weeks long
    train_end = test_start  # The training set ends right before the first test set
    train_start = 0  # The training set starts from the very beginning
    
    # Store the initial split
    train_indices = list(range(train_start, test_start_idx))
    test_indices = list(range(test_start_idx, test_start_idx + test_size))  # Regular test set
    splits.append((train_indices, test_indices))
    
    # Perform the split manually with a walking window on the test set
    while True:
        # Update the train and test indices for the next iteration
        train_end = test_start + step_size_timedelta  # The training set now includes 4 observations from the current test set (not in overlap)
        
        # Move the test set forward by test_size - overlap, while maintaining overlap
        test_start_idx += (test_size - overlap)  # Slide forward, maintaining overlap
        if test_start_idx >= len(data):
            break  # Stop when we run out of data for the test set

        test_start = data.index[test_start_idx]
        test_end = test_start + test_size_timedelta  # The test set remains of fixed size
        
        # For the last test set, ensure it has 12 weeks, even if we don't have enough data
        if test_end > data.index[-1]:
            # Calculate how many observations we have left and adjust to 12 weeks
            test_end = data.index[-1]  # Adjust to the last available observation
            test_indices = list(range(test_start_idx, len(data)))  # Take all remaining data
        else:
            test_indices = list(range(test_start_idx, test_start_idx + test_size))  # Regular 12 weeks test set
        
        # Prepare the new train indices
        train_indices = list(range(0, test_start_idx))  # The training set always starts from the beginning
        
        splits.append((train_indices, test_indices))
        
        # Break if we've covered all data points
        if test_end >= data.index[-1]:
            break
    
    return splits

    
def update_test_values(X: pd.DataFrame, y: pd.Series, day: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares the feature and target data for testing on a specific day.

    This function extracts a single row (or the remaining rows if it's the last day) 
    from the input feature DataFrame (X) and target Series (y) to create a test set for 
    a specific day. The day is specified relative to the end of the DataFrame, where
    day 1 represents the last day, day 2 the second-to-last day, and so on.

    Args:
        X (pd.DataFrame): The feature DataFrame containing all historical data.
        y (pd.Series): The target Series containing all historical target values.
        day (int): The day to extract for testing, relative to the end of the DataFrames.
                   1 is the last day, 2 is the second-to-last, etc.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X_test (pd.DataFrame): A DataFrame with the features for the specified day.
            - y_test (pd.Series): A Series with the target value for the specified day.

    Raises:
        IndexError: If the specified `day` is out of bounds for the input DataFrames.
    """
    if day != 1:
        # Select a single row using negative indexing
        X_test = X.iloc[-day:-day+1,:]
        y_test = y.iloc[-day:-day+1]

    else:
        # Handle the special case of the last day (day 1)
        X_test = X.iloc[-day:,:]
        y_test = y.iloc[-day:]

    X_test.reset_index(drop=True, inplace=True)

    return X_test, y_test


def calculate_feature_importance(model, X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the feature importance of a trained model and returns the results as a DataFrame.

    Args:
        model: A trained model object that has a `feature_importances_` attribute (e.g., scikit-learn tree-based models).
        X_train (pd.DataFrame): The training dataset used to train the model. This is used to extract feature names.

    Returns:
        pd.DataFrame: A DataFrame containing two columns:
            - 'Feature': The names of the features.
            - 'Importance': The corresponding importance scores for each feature.
    """
    if not hasattr(model, 'feature_importances_'):
        # raise AttributeError("The provided model does not have a `feature_importances_` attribute.")
        logger.warning("The provided model does not have a `feature_importances_` attribute.")

        # Return a DataFrame with all zeros for feature names and importances
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': [0] * len(X_train.columns)
        })

        return feature_importance_df

    importances = model.feature_importances_
    feature_names = list(X_train.columns)

    if len(feature_names) != len(importances):
        raise ValueError("The number of features in `X_train` does not match the number of importance scores.")

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    return feature_importance_df


def get_evals_result(model, model_type: str):
    if hasattr(model, "evals_result_"):
        return model.evals_result_
    elif model_type == "catboost" and hasattr(model, "get_evals_result"):
        return model.get_evals_result()
    else:
        # raise ValueError(f"Model {model_type} não possui evals_result disponível.")
        return None



def calculate_metrics(pred_df: pd.DataFrame, actuals: str, predictions: str) -> pd.DataFrame:
    """
    Calculates evaluation metrics for a given DataFrame of predicted and actual values.

    This function calculates the Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE), 
    Mean Absolute Error (MAE), and Weighted Absolute Percentage Error (WAPE) for the given DataFrame. 
    It also calculates the bias of the predictions.

    Args:
        pred_df (pd.DataFrame): A DataFrame containing the predicted and actual values. It should have columns 
                                named as specified in the 'actuals' and 'predictions' parameters.
        actuals (str): The name of the column in 'pred_df' containing the actual values.
        predictions (str): The name of the column in 'pred_df' containing the predicted values.

    Returns:
        pd.DataFrame: The input DataFrame 'pred_df' with additional columns for each calculated metric: 
                  'MAPE', 'RMSE', 'MAE', 'WAPE', and 'Bias'.
    """
    
    logger.debug("Calculating the evaluation metrics...")
    
    model_mape = round(mean_absolute_percentage_error(pred_df[actuals], pred_df[predictions]), 4)
    model_rmse = round(root_mean_squared_error(pred_df[actuals], pred_df[predictions]), 2)
    model_mae = round(mean_absolute_error(pred_df[actuals], pred_df[predictions]), 2)
    model_wape = round((pred_df[actuals] - pred_df[predictions]).abs().sum() / pred_df[actuals].sum(), 2)
    bias = ((pred_df[predictions] - pred_df[actuals]) / (pred_df[predictions] + pred_df[actuals])).values.round(3)
    r2 = round(r2_score(pred_df[actuals], pred_df[predictions]), 4)

    model_mape_last_week = round(mean_absolute_percentage_error(pred_df[actuals].values[-1:],
        pred_df[predictions].values[-1:]), 4)
    model_rmse_last_week = round(root_mean_squared_error(pred_df[actuals].values[-1:],
        pred_df[predictions].values[-1:]), 2)

    pred_df["MAPE"] = model_mape
    pred_df["RMSE"] = model_rmse
    pred_df["MAE"] = model_mae
    pred_df["R2"] = r2
    pred_df["WAPE"] = model_wape
    pred_df["Bias"] = bias
    pred_df["MAPE_6W"] = model_mape_last_week
    pred_df["RMSE_6W"] = model_rmse_last_week

    return pred_df


def stepwise_prediction(
    X: pd.DataFrame,
    y: pd.Series,
    forecast_horizon: int,
    model_type: Any,
    load_best_params: bool = False
) -> pd.DataFrame:
    """
    Performs iterativly 1 step ahead forecast validation for a given model type and ticker symbol.

    This function iteratively trains a model on historical data, then forecasts into the future using a sliding window approach.
    The forecast horizon is adjusted to exclude weekends. It returns a DataFrame with the actual and predicted values, along with performance metrics.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        forecast_horizon (int): The number of days to forecast ahead.
        model_type (Any): The type of model to use.
        ticker (str): The stock ticker symbol.
        load_best_params (bool): If True, loads saved best model parameters; otherwise, uses default parameters. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - date: The dates of the predictions.
            - ACTUAL: The actual target values.
            - PREDICTED_COL: The predicted values.
            - MODEL_TYPE: The type of model used.
            - CLASS: "Testing" (indicates the type of data).
            - Additional columns with performance metrics (MAE, RMSE, MAPE).
    """

    # Create empty list for storing each prediction
    predictions = []
    predictions_upper = []
    predictions_lower = []
    actuals = []
    dates = []
    X_testing_df = pd.DataFrame()

    # get the one-shot training set
    X_train = X.iloc[:-forecast_horizon, :]
    y_train = y.iloc[:-forecast_horizon]
    final_y = y_train.copy()

    logger.info(f"Training model [{model_type}] up to date (including): {X_train['date'].max().date()}")

    best_model = train(
            X_train = X_train.drop(columns=["date"]),
            y_train = y_train,
            model_type = model_type,
            load_best_params = load_best_params,
    )


    if model_type == 'ARIMA':
        predictions = best_model.predict(n_periods=forecast_horizon).values
        train_mape = train_rmse = 0
        empty_shape = np.zeros((forecast_horizon, X_train.drop(columns=["date"]).shape[1]))
        feature_importance_df = pd.DataFrame(empty_shape)
        feature_importance_df['MODEL_TYPE'] = model_type

        # Run the iteration only to fill-in the info
        for day in range(forecast_horizon, 0, -1):
            X_test, y_test = update_test_values(X, y, day)
            logger.debug(f"Testing Date: {X_test['date'].min().date()}")

            # store the results
            actuals.append(y_test.values[0])
            dates.append(X_test["date"].max())
            X_testing_df = pd.concat([X_testing_df, X_test], axis=0)

    else:
        # if model_type in ["SVR", "Ridge"]:
        #     feature_importance_df = pd.DataFrame()
        # else:
        feature_importance_df = calculate_feature_importance(best_model,  X_train.drop(columns=["date"]))
        feature_importance_df['MODEL_TYPE'] = model_type

        # Predict on training to evaluate overfitting
        train_preds = best_model.predict(X_train.drop(columns=["date"]))
        train_mape = round(mean_absolute_percentage_error(y_train, train_preds), 4)
        train_rmse = round(np.sqrt(mean_squared_error(y_train, train_preds)), 2)

        # Those comments are for using Quantile Regression models
        # The predictions comes in a multi-dimensional array
        # train_mape = round(mean_absolute_percentage_error(y_train, train_preds[:,1]), 4)
        # train_rmse = round(np.sqrt(mean_squared_error(y_train, train_preds[:,1])), 2)

        for day in range(forecast_horizon, 0, -1):
            X_test, y_test = update_test_values(X, y, day)
            logger.debug(f"Testing Date: {X_test['date'].min().date()}")

            if len(predictions) != 0:

                X_test = update_target_lag_features(X_test, -1, list(final_y.values), X_test.columns)
                X_test = update_target_sma_features(X_test, -1, list(final_y.values), X_test.columns)
                X_test = update_target_ms_features(X_test, -1, list(final_y.values), X_test.columns)
                X_test = update_bollinger_bands(X_test, -1, list(final_y.values), X_test.columns)
                X_test = update_target_comparison_lag_features(X_test, -1, list(final_y.values), X_test.columns)
                # X_test = update_target_comparison_ms_features(X_test, -1, list(final_y.values), X_test.columns)
                # X_test = update_target_comparison_sma_features(X_test, -1, list(final_y.values), X_test.columns)
                # X_test = update_rsi(X_test, -1, list(final_y.values), X_test.columns)
                
            prediction = best_model.predict(X_test.drop("date", axis=1))

            # store the results
            predictions.append(prediction[0])
            # predictions_lower.append(prediction[0][0])
            # predictions_upper.append(prediction[0][2])
            actuals.append(y_test.values[0])
            dates.append(X_test["date"].max())

            final_y = pd.concat([final_y, pd.Series(prediction[0])], axis=0)
            final_y = final_y.reset_index(drop=True)
            X_testing_df = pd.concat([X_testing_df, X_test], axis=0)


    pred_df = pd.DataFrame(list(zip(dates, actuals, predictions)), columns=["date", "ACTUAL", PREDICTED_COL])
    # Quantile:
    # pred_df = pd.DataFrame(list(zip(dates, actuals, predictions, predictions_lower, predictions_upper)), columns=["date", "ACTUAL", PREDICTED_COL, f'{PREDICTED_COL}_lower', f'{PREDICTED_COL}_upper'])

    # pred_df["MODEL_TYPE"] = type(best_model).__name__
    pred_df["MODEL_TYPE"] = model_type
    pred_df["TRAINING_MAPE"] = train_mape
    pred_df["TRAINING_RMSE"] = train_rmse
    
    X_testing_df[PREDICTED_COL] = predictions[1]
    # Quantile:
    # X_testing_df[f'{PREDICTED_COL}_lower'] = predictions[0]
    # X_testing_df[f'{PREDICTED_COL}_upper'] = predictions[2]

    X_testing_df['MODEL_TYPE'] = type(best_model).__name__
    X_testing_df.reset_index(drop=True, inplace=True)

    return pred_df, X_testing_df, feature_importance_df, best_model


def predict_arima(model, forecast_horizon, test_df, predicted_col):
    """
    Performs ARIMA forecasting on a given test DataFrame.

    This function takes an ARIMA model, a forecast horizon, a test DataFrame, and a predicted column name as input.
    It uses the ARIMA model to forecast the specified number of steps into the future and appends the forecasted
    values to the test DataFrame. It also calculates and appends lower and upper confidence intervals for the forecasts.

    Args:
        model (statsmodels.tsa.arima.model.ARIMA): The trained ARIMA model to use for forecasting.
        forecast_horizon (int): The number of steps into the future to forecast.
        test_df (pd.DataFrame): The DataFrame containing the test data. It should have a column named as specified in the 'predicted_col' parameter.
        predicted_col (str): The name of the column in 'test_df' where the forecasted values will be stored.

    Returns:
        pd.DataFrame: The input 'test_df' with additional columns for the forecasted values ('predicted_col'), 
                  lower forecast confidence interval ('lower_forecast'), and upper forecast confidence interval ('upper_forecast').
    """

    predictions_df = test_df.copy()

    forecast = model.get_forecast(steps=forecast_horizon)

    predictions_df.loc[:, predicted_col] = forecast.predicted_mean.values
    predictions_df.loc[:, ['lower_forecast', 'upper_forecast']] = forecast.conf_int(alpha=0.2).values

    return predictions_df


def save_results_to_csv(
    dataframes_list: list[tuple[str, pd.DataFrame]], 
    output_dir: str,
    filename_prefix: str,
) -> None:
    """
    Saves a DataFrame to a CSV file in the specified directory.

    Args:
        dataframes_list (list[tuple[str, pd.DataFrame]]): List of (filename, dataframe) tuples.
        output_dir (str): Directory to save the CSV files.
        filename_prefix (str): Prefix to save the CSV files.

    Raises:
        ValueError: If the `output_dir` does not exist or is not a directory.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not os.path.isdir(output_dir):
        raise ValueError(f"The specified output path '{output_dir}' is not a directory.")

    for filename, df in dataframes_list:

        file_path = os.path.join(output_dir, f"{filename_prefix}{filename}.csv")
        if os.path.isfile(file_path):
            # df.to_csv(file_path, mode='a', header=False, index=False)
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)

        logger.info(f"Saved {filename} → {file_path}")

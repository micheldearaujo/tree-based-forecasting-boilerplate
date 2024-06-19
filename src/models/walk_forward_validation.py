# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import os
import yaml
import datetime as dt
import logging
import logging.config
import argparse
from dateutil.relativedelta import relativedelta
from typing import Any

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from src.models.train_model import train_model
from src.models.predict_model import update_lag_features, update_ma_features
from src.visualization.data_viz import visualize_validation_results
from src.utils import weekend_adj_forecast_horizon

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)


with open("src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())
    model_config = config['model_config']
    data_config = config['data_config']
    PROCESSED_DATA_PATH = data_config['paths']['processed_data_path']
    PROCESSED_DATA_NAME = data_config['table_names']['processed_table_name']
    OUTPUT_DATA_PATH = data_config['paths']['output_data_path']
    OUTPUT_DATA_NAME = data_config['table_names']['output_table_name']
    DAILY_PERFORMANCE_DATA_NAME = data_config['table_names']['model_performance_table_name']
    CROSS_VAL_DATA_NAME = data_config['table_names']['cross_validation_table_name']
    MODELS_PATH = data_config['paths']['models_path']
    TARGET_COL = model_config['target_col']
    CATEGORY_COL = model_config['category_col']
    PREDICTED_COL = model_config['predicted_col']
    FORECAST_HORIZON = model_config['forecast_horizon']
    features_list = config['features_list']
    available_models = model_config['available_models']


def evaluate_and_store_performance(model_type, ticker, y_true, y_pred, latest_price_date, latest_run_date):
    """Evaluates model performance for a single ticker symbol on a single day and stores results."""

    bias = ((y_pred - y_true) / (y_pred + y_true)).round(3)
    y_true = np.array([y_true])
    y_pred = np.array([y_pred])
    mae = mean_absolute_error(y_true, y_pred).round(3)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)).round(3)
    mape = mean_absolute_percentage_error(y_true, y_pred).round(3)
    
    # Prepare data for storage
    results = {
        'EVAL_DATE': latest_price_date,
        'RUN_DATE': latest_run_date,
        'MODEL_NAME': model_type.upper(),
        CATEGORY_COL: ticker,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'BIAS': bias,
        TARGET_COL: y_pred[0],
        'ACTUAL': y_true[0]
    }
    
    results_df = pd.DataFrame([results])
    
    file_path = f"{OUTPUT_DATA_PATH}/{DAILY_PERFORMANCE_DATA_NAME}"
    if os.path.isfile(file_path):
        results_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(file_path, index=False)


def daily_model_evaluation(model_type=None, ticker=None):
    """Performs the models' performance daily evaluation"""

    # Loads the out of sample forecast table and the training dataset
    current_train_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_df.csv'), parse_dates=["DATE"])
    historical_forecasts_df = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, 'forecast_output_df.csv'), parse_dates=["DATE","RUN_DATE"])

    latest_price_date = current_train_df["DATE"].max().date()
    latest_run_date = historical_forecasts_df["RUN_DATE"].max().date()

    logger.debug(f"Latest availabe date: {latest_price_date}")
    logger.debug(f"Latest run date: {latest_run_date}")

    # Check the ticker parameter
    if ticker:
        ticker = ticker.upper() + '.SA'
        historical_forecasts_df = historical_forecasts_df[historical_forecasts_df[CATEGORY_COL] == ticker]
        current_train_df = current_train_df[current_train_df[CATEGORY_COL] == ticker]
    
    # Check the model_type parameter 
    if model_type is not None and model_type not in available_models:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from: {available_models}")
    
    elif model_type:
        available_models = [model_type]

    for ticker in historical_forecasts_df[CATEGORY_COL].unique():
        ticker_hist_forecast = historical_forecasts_df[historical_forecasts_df[CATEGORY_COL] == ticker]
        ticker_train_df = current_train_df[current_train_df[CATEGORY_COL] == ticker]

        for model_type in available_models:
            model_type = model_type.upper()

            latest_value = ticker_train_df[TARGET_COL].values[-1]

            try:
                predicted_value = ticker_hist_forecast[
                    (ticker_hist_forecast["MODEL_TYPE"] == model_type) \
                    & (ticker_hist_forecast["RUN_DATE"] == pd.to_datetime(latest_run_date)) \
                    & (ticker_hist_forecast["DATE"] == pd.to_datetime(latest_price_date))
                ][PREDICTED_COL].values[0]

            except:
                logger.error(f"\nThe last forecasts where made at {latest_run_date}, A.K.A today. Comeback tomorrow in order to calculate today's performance.")
                raise ValueError(f"\nThe last forecasts where made at {latest_run_date}, A.K.A today. Comeback tomorrow in order to calculate today's performance.")

            evaluate_and_store_performance(model_type, ticker, latest_value, predicted_value, latest_price_date, latest_run_date)


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


def calculate_metrics(pred_df: pd.DataFrame, actuals, predictions):
    """Calculate evaluation metrics"""
    logger.debug("Calculating the evaluation metrics...")
    
    model_mape = round(mean_absolute_percentage_error(actuals, predictions), 4)
    model_rmse = round(np.sqrt(mean_squared_error(actuals, predictions)), 2)
    model_mae = round(mean_absolute_error(actuals, predictions), 2)
    model_wape = round((pred_df.ACTUAL - pred_df.FORECAST).abs().sum() / pred_df.ACTUAL.sum(), 2)

    pred_df["MAPE"] = model_mape
    pred_df["MAE"] = model_mae
    pred_df["WAPE"] = model_wape
    pred_df["RMSE"] = model_rmse

    return pred_df


def stepwise_prediction(X: pd.DataFrame, y: pd.Series, forecast_horizon: int, model_type: Any, ticker: str, tune_params: bool = False) -> pd.DataFrame:
    """
    Performs iterativly 1 step ahead forecast validation for a given model type and ticker symbol.

    This function iteratively trains a model on historical data, then forecasts into the future using a sliding window approach.
    The forecast horizon is adjusted to exclude weekends. It returns a DataFrame with the actual and predicted values, along with performance metrics.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        forecast_horizon (int): The number of days to forecast ahead.
        model_type (Any): The type of model to use (e.g., 'xgb', 'rf', 'et').
        ticker (str): The stock ticker symbol.
        tune_params (bool, optional): Whether to perform hyperparameter tuning. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - DATE: The dates of the predictions.
            - ACTUAL: The actual target values.
            - PREDICTED_COL: The predicted values.
            - MODEL_TYPE: The type of model used.
            - CLASS: "Testing" (indicates the type of data).
            - Additional columns with performance metrics (MAE, RMSE, MAPE).
    """

    # Create empty list for storing each prediction
    predictions = []
    actuals = []
    dates = []
    X_testing_df = pd.DataFrame()

    forecast_horizon = weekend_adj_forecast_horizon(forecast_horizon, 2)
    
    # get the one-shot training set
    X_train = X.iloc[:-forecast_horizon, :]
    y_train = y.iloc[:-forecast_horizon]
    final_y = y_train.copy()

    logger.debug(f"Last training date: {X_train["DATE"].max().date()}")

    best_model = train_model(
        X_train.drop(columns=["DATE"]),
        y_train,
        model_type,
        ticker,
        tune_params,
        save_model=False
    )

    for day in range(forecast_horizon, 0, -1):
        X_test, y_test = update_test_values(X, y, day)

        logger.debug(f"Testing Date: {X_test["DATE"].min().date()}")

        if len(predictions) != 0:

            X_test = update_lag_features(X_test, -1, list(final_y.values), X_test.columns)
            X_test = update_ma_features(X_test, -1, list(final_y.values), X_test.columns)

        prediction = best_model.predict(X_test.drop("DATE", axis=1))

        # store the results
        predictions.append(prediction[0])
        actuals.append(y_test.values[0])
        dates.append(X_test["DATE"].max())

        final_y = pd.concat([final_y, pd.Series(prediction[0])], axis=0)
        final_y = final_y.reset_index(drop=True)
        X_testing_df = pd.concat([X_testing_df, X_test], axis=0)

    pred_df = pd.DataFrame(list(zip(dates, actuals, predictions)), columns=["DATE", "ACTUAL", PREDICTED_COL])
    pred_df = calculate_metrics(pred_df, actuals, predictions)
    pred_df["MODEL_TYPE"] = str(type(best_model)).split('.')[-1][:-2]
    pred_df["CLASS"] = "Testing"
    
    X_testing_df[PREDICTED_COL] = predictions
    X_testing_df.reset_index(drop=True, inplace=True)

    # Plotting the Validation Results
    # validation_metrics_fig = visualize_validation_results(pred_df, model_mape, model_mae, model_wape, ticker)

    # Plotting the Learning Results
    #learning_curves_fig, feat_imp = extract_learning_curves(best_model, display=True)
    
    return pred_df, X_testing_df


def walk_forward_validation(tune_params, model_type, ticker, wfv_steps=0, wfv_size=FORECAST_HORIZON):
    """
    Performs Walkf Forward Validation, i.e, training and testing the models
    in multiple time-frames.
    """

    available_models = config['model_config']['available_models']

    validation_report_df = pd.DataFrame()

    logger.debug("Loading the featurized dataset..")
    feature_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_NAME), parse_dates=["DATE"])

    # Check the ticker parameter
    if ticker:
        ticker = ticker.upper() + '.SA'
        feature_df = feature_df[feature_df[CATEGORY_COL] == ticker]
        
    # Check the model_type parameter 
    if model_type is not None and model_type not in available_models:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from: {available_models}")
    
    elif model_type:
        available_models = [model_type]

    for ticker in feature_df[CATEGORY_COL].unique():
        filtered_feature_df = feature_df[feature_df[CATEGORY_COL] == ticker].copy().drop(CATEGORY_COL, axis=1)
        
        for model_type in available_models:
            logger.info(f"Performing model cross validation for ticker symbol [{ticker}] using model [{model_type}]...")

            wfv_start_date = filtered_feature_df["DATE"].max() - relativedelta(days=wfv_size*wfv_steps)
            step_df = filtered_feature_df[filtered_feature_df["DATE"] <= wfv_start_date].copy()

            logger.info(f"WFV with {wfv_steps} steps and step size equal to {wfv_size}...")
            logger.info(f"Start training date: {step_df['DATE'].max()}")

            for step in range(wfv_steps+1):
                logger.info(f"Iteration [{step}] training date: {step_df['DATE'].max()}")

                predictions_df, X_testing_df = stepwise_prediction(
                    X=step_df.drop(columns=[TARGET_COL], axis=1),
                    y=step_df[TARGET_COL],
                    forecast_horizon=FORECAST_HORIZON,
                    model_type=model_type,
                    ticker=ticker,
                    tune_params=tune_params
                )

                predictions_df[CATEGORY_COL] = ticker
                predictions_df["TRAINING_DATE"] = dt.datetime.today().date()
                validation_report_df = pd.concat([validation_report_df, predictions_df], axis=0)

                # Add the previous testing dates to the training dataset
                step_df = filtered_feature_df[filtered_feature_df["DATE"] <= (wfv_start_date + relativedelta(days=wfv_size * (step + 1)))].copy()

    
    logger.info("Writing the testing results dataframe...")
    file_path = os.path.join(OUTPUT_DATA_PATH, 'wfv_'+CROSS_VAL_DATA_NAME)
    if os.path.isfile(file_path):
        validation_report_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        validation_report_df.to_csv(file_path, index=False)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform Out-of-Sample Tree-based models Inference.")

    parser.add_argument(
        "-mt", "--model_type",
        type=str,
        choices=["XGB", "ET"],
        help="Model name use for inference (XGB, ET) (optional, defaults to all)."
    )
    parser.add_argument(
        "-ts", "--ticker",
        type=str,
        help="""Ticker Symbol for inference. (optional, defaults to all).
        Example: BOVA -> BOVA11.SA | PETR4 -> PETR4.SA"""
    )
    args = parser.parse_args()

    logger.info("Starting the Daily Model Evaluation pipeline...")
    walk_forward_validation(False, args.model_type, args.ticker)
    logger.info("Daily Model Evaluation Pipeline completed successfully!")
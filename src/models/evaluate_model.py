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
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from src.models.train_model import train_model
from src.models.predict_model import update_lag_features, update_ma_features
from src.visualization.data_viz import visualize_validation_results
from src.utils import weekend_adj_forecast_horizon

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    

with open("src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())
    PROCESSED_DATA_PATH = config['paths']['processed_data_path']
    OUTPUT_DATA_PATH = config['paths']['output_data_path']

def evaluate_and_store_performance(model_type, ticker_symbol, y_true, y_pred, latest_price_date, latest_run_date):
    """Evaluates model performance for a single ticker symbol on a single day and stores results."""
    logger.debug(f"Actual value: {y_true}. Predicted Value: {y_pred}")

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
        'TICKER': ticker_symbol,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'BIAS': bias,
        'FORECAST': y_pred[0],
        'ACTUAL': y_true[0]
    }
    
    results_df = pd.DataFrame([results])
    
    file_path = f"{OUTPUT_DATA_PATH}/model_performance_daily.csv"
    if os.path.isfile(file_path):
        results_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(file_path, index=False)


def daily_model_evaluation(model_type=None, ticker_symbol=None):
    """Performs the models' performance daily evaluation"""

    # Loads the out of sample forecast table and the training dataset
    current_train_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["DATE"])
    historical_forecasts_df = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, 'output_stock_prices.csv'), parse_dates=["DATE","RUN_DATE"])

    TARGET_NAME = config["model_config"]["TARGET_NAME"]
    PREDICTED_NAME = config["model_config"]["PREDICTED_NAME"]
    available_models = config['model_config']['available_models']

    latest_price_date = current_train_df["DATE"].max().date()
    latest_run_date = historical_forecasts_df["RUN_DATE"].max().date()

    logger.debug(f"Latest availabe date: {latest_price_date}")
    logger.debug(f"Latest run date: {latest_run_date}")

    # Check the ticker_symbol parameter
    if ticker_symbol:
        ticker_symbol = ticker_symbol.upper() + '.SA'
        historical_forecasts_df = historical_forecasts_df[historical_forecasts_df["STOCK"] == ticker_symbol]
        current_train_df = current_train_df[current_train_df["STOCK"] == ticker_symbol]
    
    # Check the model_type parameter 
    if model_type is not None and model_type not in available_models:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from: {available_models}")
    
    elif model_type:
        available_models = [model_type.upper()]

    for ticker_symbol in historical_forecasts_df["STOCK"].unique():
        ticker_hist_forecast = historical_forecasts_df[historical_forecasts_df["STOCK"] == ticker_symbol]
        ticker_train_df = current_train_df[current_train_df["STOCK"] == ticker_symbol]

        for model_type in available_models:
            model_type = model_type.upper()

            latest_value = ticker_train_df[TARGET_NAME].values[-1]

            try:
                predicted_value = ticker_hist_forecast[
                    (ticker_hist_forecast["MODEL_TYPE"] == model_type) \
                    & (ticker_hist_forecast["RUN_DATE"] == pd.to_datetime(latest_run_date)) \
                    & (ticker_hist_forecast["DATE"] == pd.to_datetime(latest_price_date))
                ][PREDICTED_NAME].values[0]

            except:
                logger.error(f"\nThe last forecasts where made at {latest_run_date}, A.K.A today. Comeback tomorrow in order to calculate today's performance.")
                raise ValueError(f"\nThe last forecasts where made at {latest_run_date}, A.K.A today. Comeback tomorrow in order to calculate today's performance.")

            evaluate_and_store_performance(model_type, ticker_symbol, latest_value, predicted_value, latest_price_date, latest_run_date)



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


def walk_forward_validation(X: pd.DataFrame, y: pd.Series, forecast_horizon: int, model_type: Any, ticker_symbol: str, tune_params: bool = False) -> pd.DataFrame:
    """
    Make predictions for the past `forecast_horizon` days using a XGBoost model.
    This model is validated using One Shot Training, it means that we train the model
    once, and them perform the `forecast_horizon` predictions only loading the mdoel.
    
    Parameters:
        X (pandas dataframe): The input data
        y (pandas Series): The target data
        forecast_horizon (int): Number of days to forecast
        
    Returns:
        pred_df: Pandas DataFrame with the forecasted values
    """
    # TODO: Continuar com a refatoração desse bloco interno

    # Create empty list for storing each prediction
    predictions = []
    actuals = []
    dates = []
    X_testing_df = pd.DataFrame()

    forecast_horizon = 3#weekend_adj_forecast_horizon(forecast_horizon, 2)
    
    # get the one-shot training set
    X_train = X.iloc[:-forecast_horizon, :]
    y_train = y.iloc[:-forecast_horizon]

    print(f"Last training date: {X_train["DATE"].max()}")
    
    final_y = y_train.copy()
    print(final_y[-2:])

    best_model = train_model(
        X_train.drop(columns=["DATE"]),
        y_train,
        model_type,
        ticker_symbol,
        tune_params,
        save_model=False
    )

    # Iterate over the dataset to perform predictions over the forecast horizon, one by one.
    # After forecasting the next step, we need to update the "lag" features with the last forecasted
    # value
    for day in range(forecast_horizon, 0, -1):
        
        X_test, y_test = update_test_values(X, y, day)

        # only the first iteration will use the true value of Close_LAG_1
        # because the following ones will use the last predicted value
        # so we simulate the process of predicting out-of-sample
        if len(predictions) != 0:
            
            # X_test2 = X_test.copy()
            lag_features = [feature for feature in X_test.columns if "LAG" in feature]
            for feature in lag_features:
                lag_value = int(feature.split("_")[-1])
                index_to_replace = list(X_test.columns).index(feature)
                # X_test.iat[0, index_to_replace] = final_y.iloc[-lag_value]
                X_test = update_lag_features(X_test, -1, list(final_y.values), X_test.columns)


            moving_averages_features = [feature for feature in X_test.columns if "MA" in feature]
            for feature in moving_averages_features:
                ma_value = int(feature.split("_")[-1])
                last_closing_princes_ma = final_y.rolling(ma_value).mean()
                last_ma = last_closing_princes_ma.values[-1]
                index_to_replace = list(X_test.columns).index(feature)
                X_test.iat[0, index_to_replace] = last_ma

            X_testing_df = pd.concat([X_testing_df, X_test], axis=0)
            
        else:
            # we jump the first iteration because we do not need to update anything.
            
            X_testing_df = pd.concat([X_testing_df, X_test], axis=0)
            pass

        # make prediction
        print(f"Testing Date: {X_test["DATE"].min()}")
        prediction = best_model.predict(X_test.drop("DATE", axis=1))
        final_y = pd.concat([final_y, pd.Series(prediction[0])], axis=0)
        final_y = final_y.reset_index(drop=True)

        # store the results
        predictions.append(prediction[0])
        actuals.append(y_test.values[0])
        dates.append(X_test["DATE"].max())

    pred_df = pd.DataFrame(list(zip(dates, actuals, predictions)), columns=["DATE", "ACTUAL", "FORECAST"])
    X_testing_df["FORECAST"] = predictions
    X_testing_df.reset_index(drop=True, inplace=True)
    print(X_testing_df)
    pred_df["FORECAST"] = pred_df["FORECAST"].astype("float64")

    logger.debug("Calculating the evaluation metrics...")
    model_mape = round(mean_absolute_percentage_error(actuals, predictions), 4)
    model_rmse = round(np.sqrt(mean_squared_error(actuals, predictions)), 2)
    model_mae = round(mean_absolute_error(actuals, predictions), 2)
    model_wape = round((pred_df.ACTUAL - pred_df.FORECAST).abs().sum() / pred_df.ACTUAL.sum(), 2)

    pred_df["MAPE"] = model_mape
    pred_df["MAE"] = model_mae
    pred_df["WAPE"] = model_wape
    pred_df["RMSE"] = model_rmse
    pred_df["MODEL"] = str(type(best_model)).split('.')[-1][:-2]


    # Plotting the Validation Results
    validation_metrics_fig = visualize_validation_results(pred_df, model_mape, model_mae, model_wape, ticker_symbol)

    # Plotting the Learning Results
    #learning_curves_fig, feat_imp = extract_learning_curves(best_model, display=True)
    
    return pred_df, X_testing_df


def model_crossval_pipeline(tune_params, model_type, ticker_symbol):

    TARGET_NAME = config["model_config"]["TARGET_NAME"]
    PREDICTED_NAME = config["model_config"]["PREDICTED_NAME"]
    FORECAST_HORIZON = config['model_config']['forecast_horizon']
    available_models = config['model_config']['available_models']

    validation_report_df = pd.DataFrame()

    logger.info("Loading the featurized dataset..")
    stock_df_feat_all = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["DATE"])

    # Check the ticker_symbol parameter
    if ticker_symbol:
        ticker_symbol = ticker_symbol.upper() + '.SA'
        stock_df_feat_all = stock_df_feat_all[stock_df_feat_all["STOCK"] == ticker_symbol]
        
    # Check the model_type parameter 
    if model_type is not None and model_type not in available_models:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from: {available_models}")
    
    elif model_type:
        available_models = [model_type.upper()]

    for ticker_symbol in stock_df_feat_all["STOCK"].unique():
        stock_df_feat = stock_df_feat_all[stock_df_feat_all["STOCK"] == ticker_symbol].copy().drop("STOCK", axis=1)
        
        for model_type in available_models:
            logger.info(f"Performing model cross validation for ticker symbol [{ticker_symbol}] using model [{model_type}]...")
 
            predictions_df, X_testing_df = walk_forward_validation(
                X=stock_df_feat.drop(columns=[TARGET_NAME], axis=1),
                y=stock_df_feat[TARGET_NAME],
                forecast_horizon=FORECAST_HORIZON,
                model_type=model_type,
                ticker_symbol=ticker_symbol,
                tune_params=tune_params
            )

            predictions_df["STOCK"] = ticker_symbol
            predictions_df["TRAINING_DATE"] = dt.datetime.today().date()

            validation_report_df = pd.concat([validation_report_df, predictions_df], axis=0)
    
    logger.info("Writing the testing results dataframe...")
    # validation_report_df = validation_report_df.rename(columns={"FORECAST": "Price"})
    validation_report_df["CLASS"] = "Testing"
    validation_report_df.to_csv(os.path.join(OUTPUT_DATA_PATH, 'validation_results_new.csv'), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform Out-of-Sample Tree-based models Inference.")

    parser.add_argument(
        "-mt", "--model_type",
        type=str,
        choices=["xgb", "et"],
        help="Model name use for inference (xgb, et) (optional, defaults to all)."
    )
    parser.add_argument(
        "-ts", "--ticker_symbol",
        type=str,
        help="""Ticker Symbol for inference. (optional, defaults to all).
        Example: bova11 -> BOVA11.SA | petr4 -> PETR4.SA"""
    )
    args = parser.parse_args()

    logger.info("Starting the Daily Model Evaluation pipeline...")
    # daily_model_evaluation(args.model_type, args.ticker_symbol)
    model_crossval_pipeline(False, args.model_type.upper(), args.ticker_symbol)
    logger.info("Daily Model Evaluation Pipeline completed successfully!")
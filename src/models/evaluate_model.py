# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import os
import yaml
import datetime as dt
import logging
import argparse
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import warnings

# from src.utils import *
from src.visualization.data_viz import visualize_validation_results

warnings.filterwarnings("ignore")

with open("src/configuration/logging_config.yaml", 'r') as f:  

    loggin_config = yaml.safe_load(f.read())
    logging.config.dictConfig(loggin_config)
    logger = logging.getLogger(__name__)

with open("src/configuration/project_config.yaml", 'r') as f:  

    config = yaml.safe_load(f.read())


PROCESSED_DATA_PATH = config['paths']['processed_data_path']
OUTPUT_DATA_PATH = config['paths']['output_data_path']


def evaluate_and_store_performance(model_type, ticker_symbol, y_true, y_pred, latest_price_date, latest_run_date):
    """Evaluates model performance for a single ticker symbol on a single day and stores results."""
    logger.debug(f"Actual value: {y_true}. Predicted Value: {y_pred}")

    # Calculate metrics
    bias = ((y_pred - y_true) / (y_pred + y_true)).round(3)

    y_true = np.array([y_true])
    y_pred = np.array([y_pred])
    mae = mean_absolute_error(y_true, y_pred).round(3)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)).round(3)
    mape = mean_absolute_percentage_error(y_true, y_pred).round(3)
    
    # Prepare data for storage
    results = {
        'EVAL_DATE': latest_price_date,#dt.datetime.today().date(),
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
        available_models = [model_type]

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
                logger.warning(f"\nThe last forecasts where made at {latest_run_date}, A.K.A today. Comeback tomorrow in order to calculate today's performance.")
                raise ValueError(f"\nThe last forecasts where made at {latest_run_date}, A.K.A today. Comeback tomorrow in order to calculate today's performance.")

            evaluate_and_store_performance(model_type, ticker_symbol, latest_value, predicted_value, latest_price_date, latest_run_date)


def cross_validation(X: pd.DataFrame, y: pd.Series, forecast_horizon: int, stock_name: str, tunning: bool = False) -> pd.DataFrame:
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

    # Create empty list for storing each prediction
    predictions = []
    actuals = []
    dates = []
    X_testing_df = pd.DataFrame()
    
    # get the one-shot training set
    X_train = X.iloc[:-forecast_horizon+2, :]
    y_train = y.iloc[:-forecast_horizon+2]

    final_y = y_train.copy()


    best_model = xgb.XGBRegressor(
            eval_metric=["rmse", "logloss"],
    )

    if tunning:
        logger.debug("HyperTunning the model...")

        best_model, results = tune_model_hyperparameters(best_model, X_train.drop(columns=["Date"]), y_train, param_distributions_dict[str(type(best_model)).split('.')[-1][:-2]], cv=5)

    else:

        logger.debug("Fitting the model without HyperTunning...")
        best_model.fit(
            X_train.drop("Date", axis=1),
            y_train,
            eval_set=[(X_train.drop("Date", axis=1), y_train)],
            verbose=10
        )

    # Iterate over the dataset to perform predictions over the forecast horizon, one by one.
    # After forecasting the next step, we need to update the "lag" features with the last forecasted
    # value
    for day in range(forecast_horizon-2, 0, -1):
        
        if day != 1:
            # the testing set will be the next day after the training and we use the complete dataset
            X_test = X.iloc[-day:-day+1,:]
            y_test = y.iloc[-day:-day+1]

        else:
            # need to change the syntax for the last day (for -1:-2 will not work)
            X_test = X.iloc[-day:,:]
            y_test = y.iloc[-day:]

        # only the first iteration will use the true value of Close_lag_1
        # because the following ones will use the last predicted value
        # so we simulate the process of predicting out-of-sample
        if len(predictions) != 0:
            
            lag_features = [feature for feature in X_test.columns if "lag" in feature]
            for feature in lag_features:
                lag_value = int(feature.split("_")[-1])
                index_to_replace = list(X_test.columns).index(feature)
                X_test.iat[0, index_to_replace] = final_y.iloc[-lag_value]


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
        prediction = best_model.predict(X_test.drop("Date", axis=1))
        final_y = pd.concat([final_y, pd.Series(prediction[0])], axis=0)
        final_y = final_y.reset_index(drop=True)

        # store the results
        predictions.append(prediction[0])
        actuals.append(y_test.values[0])
        dates.append(X_test["Date"].max())

    pred_df = pd.DataFrame(list(zip(dates, actuals, predictions)), columns=["Date", 'Actual', 'Forecast'])
    pred_df["Forecast"] = pred_df["Forecast"].astype("float64")

    logger.debug("Calculating the evaluation metrics...")
    model_mape = round(mean_absolute_percentage_error(actuals, predictions), 4)
    model_rmse = round(np.sqrt(mean_squared_error(actuals, predictions)), 2)
    model_mae = round(mean_absolute_error(actuals, predictions), 2)
    model_wape = round((pred_df.Actual - pred_df.Forecast).abs().sum() / pred_df.Actual.sum(), 2)

    pred_df["MAPE"] = model_mape
    pred_df["MAE"] = model_mae
    pred_df["WAPE"] = model_wape
    pred_df["RMSE"] = model_rmse
    pred_df["Model"] = str(type(best_model)).split('.')[-1][:-2]

    # Plotting the Validation Results
    validation_metrics_fig = visualize_validation_results(pred_df, model_mape, model_mae, model_wape, stock_name)

    # Plotting the Learning Results
    #learning_curves_fig, feat_imp = extract_learning_curves(best_model, display=True)
    
    return pred_df, X_testing_df


def model_evaluation_pipeline(tunning=False):

    logger.debug("Loading the featurized dataset..")
    stock_df_feat_all = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])

    # iterate over the stocks
    validation_report_df = pd.DataFrame()

    for stock_name in stock_df_feat_all["Stock"].unique():

        logger.info("Testing the model for the stock: %s..."%stock_name)
        stock_df_feat = stock_df_feat_all[stock_df_feat_all["Stock"] == stock_name].copy().drop("Stock", axis=1)
        
        predictions_df, X_testing_df = test_model_one_shot(
            X=stock_df_feat.drop([model_config["TARGET_NAME"]], axis=1),
            y=stock_df_feat[model_config["TARGET_NAME"]],
            forecast_horizon=model_config['FORECAST_HORIZON'],
            stock_name=stock_name,
            tunning=tunning
        )

        predictions_df["Stock"] = stock_name
        predictions_df["Training_Date"] = dt.datetime.today().date()

        validation_report_df = pd.concat([validation_report_df, predictions_df], axis=0)
    
    logger.debug("Writing the testing results dataframe...")
    validation_report_df = validation_report_df.rename(columns={"Forecast": "Price"})
    validation_report_df["Class"] = "Testing"

    validation_report_df.to_csv(os.path.join(OUTPUT_DATA_PATH, 'validation_stock_prices.csv'), index=False)


# Execute the whole pipeline
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
    daily_model_evaluation(args.model_type, args.ticker_symbol)
    logger.info("Daily Model Evaluation Pipeline completed successfully!")
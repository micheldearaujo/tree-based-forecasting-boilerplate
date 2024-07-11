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

from src.models.evaluate_model import *


def walk_forward_validation(load_best_params, models_list, ticker_list, wfv_steps=0, wfv_size=FORECAST_HORIZON, write_to_table=True) -> pd.DataFrame:
    """
    Performs Walk Forward Validation (WFV) for your forecasting models.

    WFV involves iteratively training and testing models on expanding time windows to simulate real-world forecasting scenarios.
    This function evaluates the performance of specified models on multiple time frames for given stock tickers.

    Args:
        load_best_params (bool): If True, loads saved best model parameters; otherwise, uses default parameters.
        models_list (list): List of model types to evaluate (e.g., ["LinearRegression", "RandomForest"]).
        ticker_list (list): List of stock ticker symbols for validation.
        wfv_steps (int, optional): Number of validation steps (windows). Defaults to 0.
        wfv_size (int, optional): Size of each validation window (in days). Defaults to FORECAST_HORIZON.
        write_to_table (bool, optional): If True, writes results to a CSV file. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing detailed validation results for each model, ticker, and window, including predictions, actual values, calculated metrics, and metadata.

    Process:
        1. Loads featurized dataset.
        2. Iterates through each ticker symbol in `ticker_list`.
        3. Iterates through each model type in `models_list`.
        4. Calculates the start date for WFV based on `wfv_steps` and `wfv_size`.
        5. Performs `wfv_steps` iterations:
            a. Filters data up to the current window's end date.
            b. Calls `stepwise_prediction` to train the model and generate predictions.
            c. Calculates performance metrics (e.g., using `calculate_metrics`).
            d. Appends results to `validation_report_df` with metadata.
            e. Expands the training data window for the next step.
        6. (Optional) Appends results to a CSV file if `write_to_table` is True. 
    """

    validation_report_df = pd.DataFrame()

    logger.debug("Loading the featurized dataset..")
    feature_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_NAME), parse_dates=["DATE"])


    for ticker in ticker_list:
        filtered_feature_df = feature_df[feature_df[CATEGORY_COL] == ticker].copy().drop(CATEGORY_COL, axis=1)
        
        for model_type in models_list:
            logger.info(f"Performing model cross validation for ticker symbol [{ticker}] using model [{model_type}]...")

            wfv_start_date = filtered_feature_df["DATE"].max() - relativedelta(days=wfv_size*wfv_steps)
            step_df = filtered_feature_df[filtered_feature_df["DATE"] <= wfv_start_date].copy()

            logger.info(f"WFV with {wfv_steps} steps and step size equal to {wfv_size}...")
            logger.info(f"Start training date: {step_df['DATE'].max()}")

            for step in range(wfv_steps+1):
                logger.info(f"Iteration [{step}] training date: {step_df['DATE'].max()}")

                predictions_df, X_testing_df = stepwise_prediction(
                    X=step_df.drop(columns=[TARGET_COL]),
                    y=step_df[TARGET_COL],
                    forecast_horizon=FORECAST_HORIZON,
                    model_type=model_type,
                    ticker=ticker,
                    load_best_params=load_best_params
                )

                predictions_df[CATEGORY_COL] = ticker
                predictions_df = calculate_metrics(predictions_df, 'ACTUAL', PREDICTED_COL)
                predictions_df["CLASS"] = "Testing"
                predictions_df["TRAINING_DATE"] = dt.datetime.today().date()
                validation_report_df = pd.concat([validation_report_df, predictions_df], axis=0)

                # Add the previous testing dates to the training dataset
                step_df = filtered_feature_df[filtered_feature_df["DATE"] <= (wfv_start_date + relativedelta(days=wfv_size * (step + 1)))].copy()

    
    if write_to_table:
        logger.info("Writing the testing results dataframe...")
        file_path = os.path.join(OUTPUT_DATA_PATH, 'default_'+CROSS_VAL_DATA_NAME)

        if os.path.isfile(file_path):
            validation_report_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            validation_report_df.to_csv(file_path, index=False)

    return validation_report_df



if __name__ == "__main__":

    validation_report_df = walk_forward_validation(
        load_best_params = True,
        models_list = model_config["available_models"],
        ticker_list = data_config["ticker_list"],
        wfv_steps = WFV_STEPS,
        wfv_size = FORECAST_HORIZON
    )

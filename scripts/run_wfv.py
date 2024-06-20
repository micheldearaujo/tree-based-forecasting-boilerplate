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


def walk_forward_validation(tune_params, models_list, ticker_list, wfv_steps=0, wfv_size=FORECAST_HORIZON):
    """
    Performs Walkf Forward Validation, i.e, training and testing the models
    in multiple time-frames.
    """

    validation_report_df = pd.DataFrame()

    logger.debug("Loading the featurized dataset..")
    feature_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_NAME), parse_dates=["DATE"])


    for ticker in ticker_list:#feature_df[CATEGORY_COL].unique():
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

    walk_forward_validation(
        tune_params = model_config["tune_params"],
        models_list = model_config["available_models"],
        ticker_list = data_config["ticker_list"],
        wfv_steps = WFV_STEPS,
        wfv_size = FORECAST_HORIZON
    )

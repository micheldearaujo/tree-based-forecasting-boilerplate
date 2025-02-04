# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import os
import warnings
import yaml
import datetime as dt
import logging
import logging.config
import argparse
from dateutil.relativedelta import relativedelta
from typing import Any

import pandas as pd
import numpy as np
import mlflow.sklearn
from mlflow.models import infer_signature
import shap

from src.models.evaluate import *
from src.models.evaluate import stepwise_prediction
from src.visualization.data_viz import *
from src.models.train import *
from src.models.predict import forecast_external_features
from src.models.evaluate import CustomTimeSeriesSplit, save_results_to_csv
import src.features.feat_eng as fe
from src.data.data_loader import load_and_preprocess_model_dataset
from src.utils.decorators import time_logger
from src.models.mlflow import log_metrics_and_artifacts

warnings.filterwarnings("ignore")

# Load logging configuration
with open("src/configuration/logging_config.yaml", 'r') as f:
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("git.cmd").setLevel(logging.WARNING)
    logging.getLogger("LightGBM").setLevel(logging.CRITICAL)
    logging.getLogger("lightgbm").setLevel(logging.CRITICAL)
    logger = logging.getLogger(__name__)



@time_logger
def walk_forward_validation_ml(
    model_df: pd.DataFrame,
    test_start_date: str,
    step_size: int,
    write_to_table: bool,
    run_name: str = 'your_run_name',
    table_name: str = 'validation_results',
    run_description: str = 'your_desc'
) -> pd.DataFrame:
    """
    Performs Walk Forward Validation (WFV) for your forecasting models.

    WFV involves iteratively training and testing models on expanding time windows to simulate real-world forecasting scenarios.
    This function evaluates the performance of specified models on multiple time frames for given stock tickers.

    Args:
        model_df (pd.DataFrame): The dataset containing features and target for validation.
        test_start_date (str): The start date for the test set.
        step_size (int): The step size for the validation windows.
        write_to_table (bool): If True, writes results to a CSV file.
        run_name (str): The name of the MLflow run.
        table_name (str): The name of the table to write results to.
        run_description (str): A description of the run.

    Returns:
        pd.DataFrame: A DataFrame containing detailed validation results for each model, ticker, and window, including predictions, actual values, calculated metrics, and metadata.
    """
    TARGET_COL = model_config["target_col"]
    PREDICTED_COL = model_config["predicted_col"]
    FORECAST_HORIZON = model_config["forecast_horizon"]
    USE_TUNED_PARMS = model_config["use_tuned_params"]

    validation_report_df = pd.DataFrame()
    feature_importance_df = pd.DataFrame()
    X_test_efective = pd.DataFrame()
    run_models = {}
    X_trains = pd.DataFrame()
    shap_values_test_df = pd.DataFrame()
    shap_values_train_df = pd.DataFrame()
    learning_curves_figs = {}
    crossval_results_figs = {}
    feat_importance_figs = {}
    summary_figs = {}

    n_iteration = 1

    tscv = CustomTimeSeriesSplit(
        data = model_df,
        test_start_date = test_start_date,
        test_size = FORECAST_HORIZON,
        step_size = step_size
    )

    logger.warning(f"Running Walk Forward Validation with [{len(tscv)}] steps and step size equal to [{step_size}]...")

    for train_index, test_index in tscv:
        X_train, X_test = model_df.drop(columns=[TARGET_COL]).iloc[train_index].copy(), \
                          model_df.drop(columns=[TARGET_COL]).iloc[test_index].copy()
        y_train, y_test = model_df.iloc[train_index][TARGET_COL].copy(), \
                          model_df.iloc[test_index][TARGET_COL].copy()

        logger.info(f"Iteration [{n_iteration}] out of [{len(tscv)}] end training date: {X_train.index.max()}...")

        window_train_date_max = X_train.index.max()

        X_train_exog_preds = forecast_external_features(lags_exog_dict, X_train, X_test, FORECAST_HORIZON)
        X_train = pd.concat([X_train, X_train_exog_preds])

        feature_df = build_features(X_train, y_train)
        reduced_df = feature_df.copy()

        X_train = reduced_df[reduced_df['date'] <= window_train_date_max]
        X_test = reduced_df[reduced_df['date'] > window_train_date_max]

        if n_iteration == 1:
            logger.debug(f"\nModel Features: {list(X_train.columns)}\n")

        models = []

        for model_type in models_list:
            logger.info(f"Performing cross-validation for [{model_type}]...")

            predictions_df, X_testing_df, feat_imp_df, best_model = stepwise_prediction(
                X = pd.concat([X_train, X_test], axis=0),
                y = pd.concat([y_train, y_test], axis=0),
                forecast_horizon = FORECAST_HORIZON,
                model_type = model_type,
                load_best_params = USE_TUNED_PARMS
            )

            predictions_df = calculate_metrics(predictions_df, "ACTUAL", PREDICTED_COL)
            feat_imp_df['WINDOW'] = n_iteration
            X_testing_df['WINDOW'] = n_iteration

            predictions_df['CLASS'] = 'validation'
            predictions_df['WINDOW'] = n_iteration
            predictions_df['TRAINING_DATE'] = dt.datetime.today()
            predictions_df['REFERENCE_DATE'] = window_train_date_max
            predictions_df['REFERENCE_PIX'] = y_train.values[-1]

            validation_report_df = pd.concat([validation_report_df, predictions_df], axis=0)
            feature_importance_df = pd.concat([feature_importance_df, feat_imp_df], axis=0)
            X_test_efective = pd.concat([X_test_efective, X_testing_df], axis=0)
            models.append(best_model)

            shap_values_test, shap_test_df = calculate_shap_values_to_df(best_model, X_testing_df.drop(columns=['date', 'MODEL_TYPE', 'WINDOW', 'FORECAST']))
            shap_test_df['WINDOW'] = n_iteration
            shap_test_df['MODEL_TYPE'] = model_type
            shap_values_test_df = pd.concat([shap_values_test_df, shap_test_df], axis=0)

            shap_values_train, shap_train_df = calculate_shap_values_to_df(best_model, X_train.drop(columns=['date']))
            shap_train_df['WINDOW'] = n_iteration
            shap_train_df['MODEL_TYPE'] = model_type
            shap_values_train_df = pd.concat([shap_values_train_df, shap_train_df], axis=0)

            ##### Figures - Only plot at the end of all iterations
            if n_iteration == len(tscv):
                fig1 = plot_crossval_results(
                    validation_report_df[validation_report_df["MODEL_TYPE"] == model_type],
                    model_type)
                crossval_results_figs[model_type] = fig1

                fig2 = plot_tree_feature_importance(
                    feature_importance_df[feature_importance_df["MODEL_TYPE"] == model_type],
                    kind='boxplot')
                feat_importance_figs[model_type] = fig2

                lc = get_evals_result(best_model, model_type)
                if lc:
                    lc_fig = plot_learning_curve(best_model, lc, model_type)
                    learning_curves_figs[model_type] = lc_fig
                else:
                    learning_curves_figs[model_type] = None

                summary_fig = plot_shap_summary(shap_values_train, X_train.drop(columns=['date']))
                summary_figs[model_type] = summary_fig

                

        X_train['WINDOW'] = n_iteration
        X_trains = pd.concat([X_trains, X_train], axis=0)
        run_models[n_iteration] = models
        n_iteration += 1

    log_metrics_and_artifacts(
        validation_report_df,
        feature_importance_df,
        X_test_efective,
        X_trains,
        shap_values_train_df,
        shap_values_test_df,
        model_df,
        run_name,
        run_description,
        crossval_results_figs,
        feat_importance_figs,
        learning_curves_figs,
        summary_figs
    )

    if write_to_table:
        logger.info("Writing the testing results dataframe...")

        OUTPUT_DIR = os.path.join(OUTPUT_DATA_PATH, 'cross-validation')

        save_results_to_csv(
            dataframes_list = [
                ("validation_report_df" ,validation_report_df),
                ("shap_values_train_df", shap_values_train_df),
                ("shap_values_test_df", shap_values_test_df),
                ("feature_importance_df", feature_importance_df)
            ],
            output_dir = OUTPUT_DIR,
            filename_prefix = table_name
        )

    return validation_report_df, X_test_efective

if __name__ == "__main__":
    feature_df = load_and_preprocess_model_dataset("featurized_df")
    feature_df = feature_df.set_index("date")
    desc = ""

    validation_report_df, _ = walk_forward_validation_ml(
        model_df        = feature_df,
        test_start_date = model_config["tuning_holdout_date"],
        step_size       = model_config["cross_validation_step_size"],
        run_name        = f"{desc}",
        table_name      = f"{desc}",
        write_to_table  = True,
        run_description = """
                        Testing the hyper parameter tuning using HyperOPT
                        instead of RandomSearch or GridSearchCV.

        {}""".format(
            "\n".join(list(lags_exog_dict.keys()))
        ),
    )
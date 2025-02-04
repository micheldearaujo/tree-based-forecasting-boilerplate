# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import os
import yaml
import datetime as dt
import logging
import logging.config
import argparse
import locale
from dateutil.relativedelta import relativedelta
from typing import Any

import pandas as pd
import numpy as np
from mlflow import MlflowClient

from src.models.evaluate import *
from src.data.data_loader import load_and_preprocess_model_dataset
from src.features.feat_eng import build_features
from src.models.train import (
    split_feat_df_Xy,
    configure_mlflow_experiment,
    get_models
)
from src.models.train import *
from src.models.predict import *
from src.configuration.config_model import model_config
from src.utils.decorators import time_logger
from scripts.run_cross_validation import walk_forward_validation_ml
from src.visualization.data_viz import (
    plot_shap_summary,
    plot_tree_feature_importance,
    plot_learning_curve,
    plot_multiple_models_forecast
)


with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)


def select_best_model(model_df: pd.DataFrame, test_start_date: str, step_size: int, metric='RMSE'):
    """
    Evaluates multiple models, selects the best based on a given metric, and stages it to "prod".

    Args:
        models (dict): A dictionary of models with their names as keys (e.g., {'XGB': xgb_model, 'ET': et_model, ...}).
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test target values.
        metric (str, optional): Evaluation metric ('rmse' or 'mae'). Defaults to 'rmse'.
    
    Returns:
        str: The name of the best model selected for this trial.
    """
    validation_report_df, _ = walk_forward_validation_ml(
        model_df        = model_df,
        test_start_date = test_start_date,
        step_size       = step_size,
        run_name        = f"best_model_trial",
        write_to_table  = False,
    )
    metrics_summary = (
        validation_report_df
        .groupby("MODEL_TYPE")[["RMSE", "MAPE"]]
        .mean()
        .reset_index()
    )

    best_model_type = metrics_summary.sort_values(metric).iloc[0]["MODEL_TYPE"]
    logger.info(f"Best model for this trial: {best_model_type}.")

    return best_model_type

@time_logger
def training_pipeline(
    model_df: pd.DataFrame,
    write_to_table: bool,
    log_to_experiment: bool
) -> tuple:
    """
    This function is responsible for training multiple forecasting models on a given dataset.
    It prepares the data, trains the models, logs the training process using MLflow, and registers the models.

    Args:
        model_df (pd.DataFrame): The dataset containing the features and target variable for training.

    Returns:
        dict: Dictionary with all trained models.
    """
    logger.info("Starting the training pipeline...")

    FORECAST_HORIZON    = model_config["forecast_horizon"]
    MODEL_NAME          = model_config["model_name"]
    EXPERIMENT_PATH     = model_config["mlflow_experiment_path"]
    EXPERIMENT_PROD_PATH= model_config["mlflow_experiment_path_production"]
    USE_TUNED_PARMS     = model_config["use_tuned_params"]
    STEP_SIZE           = model_config["cross_validation_step_size"]
    MODEL_SEL_ENGINE    = model_config["model_selection_engine"]
    models_list         = list(get_models().keys())

    predictions_df = pd.DataFrame()
    full_shap_values = pd.DataFrame()

    client = MlflowClient()
    X_train, y_train = model_df.drop(columns=[TARGET_COL]).copy(), model_df[TARGET_COL].copy()

    X_train = build_features(X_train, y_train)
    logger.info(f"Most recent date on training data: {X_train['date'].max()}")
    logger.debug(f"Features to be trained on: {list(X_train.columns)}")

    today_year = dt.datetime.today().year
    today_month = dt.datetime.today().month

    TRIAL_DATE = X_train.iloc[: -(FORECAST_HORIZON-1)]["date"].max()

    if MODEL_SEL_ENGINE == 'best':
        best_model_type = select_best_model(
            model_df        = model_df,
            test_start_date = TRIAL_DATE,
            step_size       = STEP_SIZE,
            metric          = "RMSE",
        )

        models_list = [best_model_type]

    elif MODEL_SEL_ENGINE == 'ensemble':
        models_list = ['ensemble']

    elif MODEL_SEL_ENGINE == 'all':
        pass

    else:
        raise ValueError("Invalid model selection engine. Choose between 'best', 'ensemble', or 'all'.")

    fitted_models = {}
    for model_type in models_list:
        mlflow.end_run()
        configure_mlflow_experiment(experiments_path=EXPERIMENT_PROD_PATH)

        with mlflow.start_run(run_name=f"best_model_{model_type}_{model_flavor}_{today_month:02d}_{today_year}") as run:
            mlflow.set_tag(
                "mlflow.note.content",
                f"""Last available date for training: {X_train['date'].max()}.
                Model selection engine: {MODEL_SEL_ENGINE}.
                Best Model: {model_type}.
                Run datetime: {dt.datetime.now()}.
                """
            )

            logger.info(f"Training model [{model_type}] of flavor [{model_flavor}]...")

            model = train(
                X_train = X_train.drop(columns=['date']),
                y_train = y_train,
                model_type = model_type,
                load_best_params = USE_TUNED_PARMS,
            )

            fitted_models[model_type] = model

            if model_type == 'ARIMA':
                fig = model.plot_diagnostics()
                mlflow.log_figure(fig, f"{model_type}_diagnostics.png")
                mlflow.log_params(model.get_params())

            logger.debug(f"Logging the model to MLflow...")

            model_signature = infer_signature(
                X_train.drop(columns=['date']).tail(),
                pd.DataFrame(y_train).tail()
            )

            shap_values, shap_values_df = calculate_shap_values_to_df(model, X_train.drop(columns=['date']))
            shap_values_df['MODEL_TYPE'] = model_type
            summary_fig = plot_shap_summary(shap_values, X_train.drop(columns=['date']))
            mlflow.log_figure(summary_fig, f"shap_summary_plot_{model_type}.png")

            feature_importance_df = calculate_feature_importance(model,  X_train.drop(columns=["date"]))
            feature_importance_df['MODEL_TYPE'] = model_type
            feat_imp_fig = plot_tree_feature_importance(
                feature_importance_df,
                kind='boxplot'
            )
            mlflow.log_figure(feat_imp_fig, f"feature_importance_{model_type}.png")

            lc = get_evals_result(model, model_type)
            if lc:
                lc_fig = plot_learning_curve(model, lc, model_type)
                mlflow.log_figure(lc_fig, f"learning_curve_{model_type}.png")

            features_list = X_train.drop(columns=['date']).columns
            mlflow.log_text("\n".join(list(features_list)), 'features_list.txt')

            model_df.to_csv('./models/processed_df.csv', index=False)
            mlflow.log_artifact('./models/processed_df.csv', artifact_path="data")
            os.remove('./models/processed_df.csv')

            X_train.to_csv('./models/X_train.csv', index=False)
            mlflow.log_artifact('./models/X_train.csv', artifact_path="data")
            os.remove('./models/X_train.csv')

            mlflow.log_params(model.get_params())

            ARTIFACT_PATH = f'{MODEL_NAME}_{model_flavor}'
            mlflow.sklearn.log_model(
                sk_model = model,
                artifact_path = ARTIFACT_PATH,
                signature = model_signature,
                input_example = X_train.drop(columns=['date']).sample(5)
            )

            logger.info("Registering the model to MLflow...")
            logger.info(f"Model registration URI: runs:/{run.info.run_id}/{run.info.run_name}")
            model_details = mlflow.register_model(
                model_uri = f"runs:/{run.info.run_id}/{run.info.run_name}",
                name = ARTIFACT_PATH
            )
            print(model_details)

            client.update_model_version(
                name = ARTIFACT_PATH,
                version = model_details.version,
                description=f"""
                This run was generated using the model selection method: {MODEL_SEL_ENGINE}.
                The best model for this run was: {model_type}.
                """
            )

            client.set_registered_model_alias(
                name = ARTIFACT_PATH,
                alias = "champion",
                version = model_details.version
            )
            client.set_model_version_tag(
                name = ARTIFACT_PATH,
                version = str(model_details.version),
                key = "hyperparam_tuned",
                value = USE_TUNED_PARMS
            )
            client.set_model_version_tag(
                name = ARTIFACT_PATH,
                version = str(model_details.version),
                key = "model_selection_engine",
                value = MODEL_SEL_ENGINE
            )


            ##### Run the out-of-sample prediction in the same pipeline
            logger.debug("Creating the future dataframe...")
            future_df = make_future_df(FORECAST_HORIZON, model_df)

            logger.debug("Predicting iteratively...")
            model_pred_df, shap_values_inference, final_shap_values_test = make_iterative_predictions(
                model=model,
                future_df=future_df,
                past_target_values=list(model_df[TARGET_COL].values),
            )

            model_pred_df["PIX_REFERENECE"] = model_df[TARGET_COL].values[
                -1
            ]
            model_pred_df["MODEL_TYPE"] = model_type
            model_pred_df["MODEL_FLAVOR"] = model_flavor
            model_pred_df["RUN_DATE"] = str(dt.datetime.today().date())
            model_pred_df["RUN_DATETIME"] =str(dt.datetime.today())
            model_pred_df["MODEL_URI"] = f"runs:/{run.info.run_id}/{run.info.run_name}"            
            predictions_df = pd.concat([predictions_df, model_pred_df], axis=0)

            shap_values_inference['MODEL_TYPE'] = model_type
            shap_values_inference["MODEL_FLAVOR"] = model_flavor
            shap_values_inference['RUN_DATE'] = str(dt.datetime.today().date())
            shap_values_inference['RUN_DATETIME'] = str(dt.datetime.today())
            shap_values_inference['MODEL_URI'] = f"runs:/{run.info.run_id}/{run.info.run_name}"
            full_shap_values = pd.concat([full_shap_values, shap_values_inference], axis=0)

            if log_to_experiment:

                fig = plot_multiple_models_forecast(
                    training_dataframe=model_df,
                    forecast_dataframe=predictions_df,
                    x_col="date",
                    y_col=[TARGET_COL, PREDICTED_COL],
                    title=f"Forecast for the next periods.<br>Inference date: {predictions_df['RUN_DATE'].unique()[0]}",
                    x_label="Date",
                    y_label="Target and Predicted Values"
                )
                mlflow.log_figure(fig, f"forecast_result_{model_type}.png")

                summary_test_fig = plot_shap_summary(final_shap_values_test, future_df.drop(columns=['date']))
                mlflow.log_figure(summary_test_fig, f"shap_summary_plot_test_{model_type}.png")

                model_pred_df.to_csv(f"./models/{model_type}_{model_flavor}_pred_df.csv", index=False)
                mlflow.log_artifact(f"./models/{model_type}_{model_flavor}_pred_df.csv", artifact_path="data")
                os.remove(f"./models/{model_type}_{model_flavor}_pred_df.csv")

                shap_values_inference.to_csv(f"./models/shap_inf_{model_type}_{model_flavor}_df.csv", index=False)
                mlflow.log_artifact(f"./models/shap_inf_{model_type}_{model_flavor}_df.csv", artifact_path="data")
                os.remove(f"./models/shap_inf_{model_type}_{model_flavor}_df.csv")


    if write_to_table:
        logger.info("🤸 Writing the predictions to database... 🥰")

        OUTPUT_PATH = './data/output/inference'
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        predictions_file_path = f"{OUTPUT_PATH}/out-of-sample-forecast_historical_{model_flavor}.csv"
        shap_file_path = f"{OUTPUT_PATH}/shap_values_historical_{model_flavor}.csv"

        if os.path.isfile(predictions_file_path):
            predictions_df.to_csv(predictions_file_path, mode="a", header=False, index=False)
        else:
            predictions_df.to_csv(predictions_file_path, index=False)
        
        if os.path.isfile(shap_file_path):
            full_shap_values.to_csv(shap_file_path, mode="a", header=False, index=False)
        else:
            full_shap_values.to_csv(shap_file_path, index=False)

        logger.info("🏍️ Predictions written successfully!🤯")

    logger.info("🚀🚀 Inference pipeline finished successfully! 🔥🔥")


    return fitted_models, predictions_df, future_df, full_shap_values


if __name__ == "__main__":

    feature_df = load_and_preprocess_model_dataset("featurized_df")
    feature_df = feature_df.set_index("date")

    trained_models = training_pipeline(
        model_df = feature_df,
        write_to_table = True,
        log_to_experiment = True
    )
    
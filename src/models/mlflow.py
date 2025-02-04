# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import pandas as pd
import pandas_gbq

import os
import warnings
import yaml
import setuptools
import datetime as dt
import time
import logging
import logging.config
import argparse
from dateutil.relativedelta import relativedelta
from typing import Any

import pandas as pd
import numpy as np
import pmdarima as pm

import mlflow.sklearn
from mlflow.models import infer_signature
import shap

from src.models.evaluate import *
from src.models.evaluate import stepwise_prediction
from src.visualization.data_viz import *
from src.models.train import *
from src.models.predict import forecast_external_features
from src.models.evaluate import CustomTimeSeriesSplit
import src.features.feat_eng as fe
from src.data.data_loader import load_and_preprocess_model_dataset
from src.utils.decorators import time_logger
from src.visualization.data_viz import plot_crossval_results, extract_learning_curves, plot_tree_feature_importance



def log_metrics_and_artifacts(
    validation_report_df,
    feature_importance_df,
    X_test_efective,
    X_trains, 
    shap_values_train, 
    shap_values_test, 
    model_df, 
    run_name, 
    run_description,
    crossval_results_figs,
    feat_importance_figs,
    learning_curves_figs,
    summary_figs
) -> None:
    """Logs metrics and artifacts to MLflow."""

    configure_mlflow_experiment()
    with mlflow.start_run(run_name=f"{run_name}_agg") as run:
        mlflow.set_tag("mlflow.note.content", run_description)

        for model_type in validation_report_df['MODEL_TYPE'].unique():
            with mlflow.start_run(run_name=f"{model_type}", nested=True) as subrun:
                mlflow.set_tag("mlflow.note.content", run_description)

                predictions_model_df = validation_report_df[validation_report_df['MODEL_TYPE'] == model_type]
                feat_imp_model_df = feature_importance_df[feature_importance_df['MODEL_TYPE'] == model_type]
                X_test_efective_model_df = X_test_efective[X_test_efective['MODEL_TYPE'] == model_type]

                shap_train_df = shap_values_train[shap_values_train['MODEL_TYPE'] == model_type].copy()
                shap_test_df = shap_values_test[shap_values_test['MODEL_TYPE'] == model_type].copy()

                # TODO: Transfer this plots to outside this function
                # because we may want to plot it even without MLFlow
                # fig1 = plot_crossval_results(predictions_model_df, model_type)
                # fig2 = plot_tree_feature_importance(feat_imp_model_df, kind='boxplot')

                mlflow.log_metric("testing_mape", round(predictions_model_df['MAPE'].mean(), 4))
                mlflow.log_metric("testing_rmse", round(predictions_model_df['RMSE'].mean(), 4))
                mlflow.log_metric("testing_mape_6w", round(predictions_model_df['MAPE_6W'].mean(), 4))
                mlflow.log_metric("testing_rmse_6w", round(predictions_model_df['RMSE_6W'].mean(), 4))
                mlflow.log_metric("training_mape", round(predictions_model_df['TRAINING_MAPE'].mean(), 4))
                mlflow.log_metric("training_rmse", round(predictions_model_df['TRAINING_RMSE'].mean(), 4))
                # mlflow.log_figure(fig1, f"validation_results_{model_type}.png")
                # mlflow.log_figure(fig2, f"feature_importance_{model_type}.png")
                mlflow.log_figure(crossval_results_figs[model_type], f"validation_results_{model_type}.png")
                mlflow.log_figure(feat_importance_figs[model_type], f"feature_importance_{model_type}.png")
                mlflow.log_figure(summary_figs[model_type], f"shap_summary_plot_{model_type}.png")
                try:
                    mlflow.log_figure(learning_curves_figs[model_type], f"learning_curve_{model_type}.png")
                except:
                    pass

        # fig1 = plot_crossval_results(validation_report_df, 'All Models')
        fig1 = plot_crossval_results(validation_report_df, 'All Models')
        fig2 = plot_tree_feature_importance(feature_importance_df, kind='boxplot')
        mlflow.log_figure(fig1, f"validation_results_general.png")
        mlflow.log_figure(fig2, f"feature_importance_general.png")
        mlflow.log_metric("avg_testing_mape", round(validation_report_df['MAPE'].mean(), 4))
        mlflow.log_metric("avg_testing_rmse", round(validation_report_df['RMSE'].mean(), 4))
        mlflow.log_metric("avg_testing_mape_6w", round(validation_report_df['MAPE_6W'].mean(), 4))
        mlflow.log_metric("avg_testing_rmse_6w", round(validation_report_df['RMSE_6W'].mean(), 4))
        mlflow.log_metric("avg_training_mape", round(validation_report_df['TRAINING_MAPE'].mean(), 4))
        mlflow.log_metric("avg_training_rmse", round(validation_report_df['TRAINING_RMSE'].mean(), 4))
        features_list = X_trains.drop(columns=['date', 'WINDOW']).columns
        mlflow.log_text("\n".join(list(features_list)), 'features_list.txt')

        # Logging the tables
        validation_report_df.to_csv('./models/validation_report_df.csv', index=False)
        mlflow.log_artifact('./models/validation_report_df.csv', artifact_path="data")
        os.remove('./models/validation_report_df.csv')

        feature_importance_df.to_csv('./models/feature_importance_df.csv', index=False)
        mlflow.log_artifact('./models/feature_importance_df.csv', artifact_path="data")
        os.remove('./models/feature_importance_df.csv')

        X_test_efective.to_csv('./models/X_test_efective.csv', index=False)
        mlflow.log_artifact('./models/X_test_efective.csv', artifact_path="data")
        os.remove('./models/X_test_efective.csv')

        X_trains.to_csv('./models/X_trains.csv', index=False)
        mlflow.log_artifact('./models/X_trains.csv', artifact_path="data")
        os.remove('./models/X_trains.csv')

        shap_values_train.to_csv('./models/shap_values_train.csv', index=False)
        mlflow.log_artifact('./models/shap_values_train.csv', artifact_path="data")
        os.remove('./models/shap_values_train.csv')

        shap_values_test.to_csv('./models/shap_values_test.csv', index=False)
        mlflow.log_artifact('./models/shap_values_test.csv', artifact_path="data")
        os.remove('./models/shap_values_test.csv')

        model_df.to_csv('./models/processed_df.csv', index=False)
        mlflow.log_artifact('./models/processed_df.csv', artifact_path="data")
        os.remove('./models/processed_df.csv')
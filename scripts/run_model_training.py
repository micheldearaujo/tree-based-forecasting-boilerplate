# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')


from src.models.train_model import *


def training_pipeline(models_list: list, ticker_list: list, tune_params=False, save_model=True):
    """
    Executes the complete model training pipeline for one or multiple ticker symbols and model types.

    The pipeline performs the following steps:
    
    1. Loads the featurized dataset.
    2. Filters the dataset based on the specified ticker symbol (if provided).
    3. Trains the specified models (or all available models if none specified).
    4. Optionally performs hyperparameter tuning using RandomizedSearchCV.
    5. Optionally saves the trained models to files.

    Args:
        model_type (list): The list of models type to train.
        ticker_symbol (list): The ticker symbol to train on.
        tune_params (bool, optional): If True, perform hyperparameter tuning. Defaults to False.
        save_model (bool, optional): If True, save the trained models to files. Defaults to True.

    Raises:
        ValueError: If an invalid model type is provided.
    """
    logger.debug("Loading the featurized dataset..")
    feature_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_NAME), parse_dates=["DATE"])
    logger.info(f"Last Available Date in Training Dataset: {feature_df['DATE'].max()}")
    
    for ticker_symbol in ticker_list:#feature_df[CATEGORY_COL].unique():
        ticker_df_feat = feature_df[feature_df[CATEGORY_COL] == ticker_symbol].drop(CATEGORY_COL, axis=1).copy()
        X_train, y_train = split_feat_df_Xy(ticker_df_feat)

        for model_type in models_list:
            logger.debug(f"Training model [{model_type}] for ticker [{ticker_symbol}]...")
            xgb_model = train_model(X_train, y_train, model_type, ticker_symbol, tune_params, save_model)

            # learning_curves_fig , feat_importance_fig = extract_learning_curves(xgboost_model)


if __name__ == "__main__":


    logger.info("Starting the training pipeline...")
    training_pipeline(
        tune_params = model_config["tune_params"],
        models_list = model_config["available_models"],
        ticker_list = data_config["ticker_list"],
        save_model = model_config["save_model"]
    )
    logger.info("Training Pipeline completed successfully!")

# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')


from src.models.train_model import *


def training_pipeline_ensemble(models_list: list[str], ticker_list: list[str], load_best_params=True):
    """
    Executes the complete model training pipeline for one or multiple ticker symbols and model types.

    The pipeline performs the following steps:
    
    1. Loads the featurized dataset.
    2. Filters the dataset based on the specified ticker symbol.
    3. Trains the specified models.

    Args:
        model_type (list): The list of models type to train.
        ticker_symbol (list): The ticker symbol to train on.
        load_best_params (bool, optional): If True, Loads the best parameters saved in joblib file inside models/. Defaults to True.
    """
    logger.debug("Loading the featurized dataset..")
    feature_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_NAME), parse_dates=["DATE"])
    logger.info(f"Last available date for training: {feature_df['DATE'].max()}.")

    for ticker_symbol in ticker_list:

        ticker_df_feat = feature_df[feature_df[CATEGORY_COL] == ticker_symbol].drop(CATEGORY_COL, axis=1).copy()
        X_train, y_train = split_feat_df_Xy(ticker_df_feat)

        # Train and save all the models
        for model_type in models_list:
            logger.debug(f"Training model [{model_type}] for ticker [{ticker_symbol}]...")

            model = train_model(X_train, y_train, model_type, ticker_symbol, load_best_params=load_best_params)

            prod_model_path = os.path.join(MODELS_PATH, ticker_symbol, f'{model_type}_model.joblib')
            joblib.dump(model, prod_model_path) 

            logger.warning(f"\nSaved model '{model_type}' for Ensemble at: {prod_model_path}")

if __name__ == "__main__":


    logger.info("Starting the training pipeline...")
    training_pipeline_ensemble(
        models_list = model_config["available_models"],
        ticker_list = data_config["ticker_list"],
        load_best_params = True,
    )
    logger.info("Training Pipeline completed successfully!")

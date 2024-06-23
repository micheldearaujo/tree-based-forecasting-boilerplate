# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')


from src.models.train_model import *


def training_pipeline(models_list: list[str], ticker_list: list[str], load_best_params=True):
    """
    Executes the complete model training pipeline for one or multiple ticker symbols and model types.

    The pipeline performs the following steps:
    
    1. Loads the featurized dataset.
    2. Filters the dataset based on the specified ticker symbol (if provided).
    3. Trains the specified models (or all available models if none specified).

    Args:
        model_type (list): The list of models type to train.
        ticker_symbol (list): The ticker symbol to train on.
        load_best_params (bool, optional): If True, Loads the best parameters saved in joblib file inside models/. Defaults to True.

    Raises:
        ValueError: If an invalid model type is provided.
    """
    logger.debug("Loading the featurized dataset..")
    feature_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_NAME), parse_dates=["DATE"])
    logger.info(f"Last available date for training: {feature_df['DATE'].max()}.")

    for ticker_symbol in ticker_list:
        ticker_df_feat = feature_df[feature_df[CATEGORY_COL] == ticker_symbol].drop(CATEGORY_COL, axis=1).copy()
        hold_out_date = ticker_df_feat["DATE"].max() - relativedelta(days=FORECAST_HORIZON)
        logger.info(f"Last available date for model selection: {hold_out_date}.")

        train_df = ticker_df_feat[ticker_df_feat["DATE"] < hold_out_date].copy()
        val_df = ticker_df_feat[ticker_df_feat["DATE"] >= hold_out_date].copy()

        X_train, y_train = split_feat_df_Xy(train_df)
        X_val, y_val = split_feat_df_Xy(val_df)

        models = {}
        for model_type in models_list:
            logger.debug(f"Training model [{model_type}] for ticker [{ticker_symbol}]...")
            model = train_model(X_train, y_train, model_type, ticker_symbol, load_best_params=load_best_params)
            models[model_type] = model

        best_model_name = select_and_stage_best_model(models, X_val, y_val)
        
        # Retrain best model
        X_train, y_train = split_feat_df_Xy(ticker_df_feat)
        best_model = train_model(X_train, y_train, best_model_name, ticker_symbol, load_best_params=load_best_params)

        # Stage to "prod" (simulate by saving the model)
        prod_model_path = os.path.join(MODELS_PATH, ticker_symbol, 'prod_model.joblib')
        joblib.dump(best_model, prod_model_path) 

        logger.warning(f"\nStaged model '{best_model_name}' to production at: {prod_model_path}")

if __name__ == "__main__":


    logger.info("Starting the training pipeline...")
    training_pipeline(
        models_list = model_config["available_models"],
        ticker_list = data_config["ticker_list"],
        load_best_params = True,
    )
    logger.info("Training Pipeline completed successfully!")

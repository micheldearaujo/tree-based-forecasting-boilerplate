# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')


from src.models.train_model import *


def training_pipeline(tune_params=False, model_type=None, ticker_symbol=None, save_model=False):
    """
    Executes the complete model training pipeline for one or multiple ticker symbols and model types.

    The pipeline performs the following steps:
    
    1. Loads the featurized dataset from 'processed_stock_prices.csv'.
    2. Filters the dataset based on the specified ticker symbol (if provided).
    3. Trains the specified models (or all available models if none specified).
    4. Optionally performs hyperparameter tuning using RandomizedSearchCV.
    5. Optionally saves the trained models to files.

    Args:
        tune_params (bool, optional): If True, perform hyperparameter tuning. Defaults to False.
        model_type (str, optional): The model type to train. If None, all available models will be trained. Valid choices are defined in `model_config['available_models']`. Defaults to None.
        ticker_symbol (str, optional): The ticker symbol to train on. If None, all ticker symbols in the dataset will be used. Valid choices are defined in `data_config['tickers_list']`.  Defaults to None.
        save_model (bool, optional): If True, save the trained models to files. Defaults to False.

    Raises:
        ValueError: If an invalid model type is provided.
    """
    logger.debug("Loading the featurized dataset..")
    feature_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_NAME), parse_dates=["DATE"])
    logger.info(f"Last Available Date in Training Dataset: {feature_df['DATE'].max()}")

    # Check the ticker_symbol parameter
    if ticker_symbol:
        ticker_symbol = ticker_symbol.upper() + '.SA'
        feature_df = feature_df[feature_df[CATEGORY_COL] == ticker_symbol]

    # Check the model_type parameter
    available_models = model_config['available_models']
    if model_type is not None and model_type not in available_models:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from: {available_models}")
    
    elif model_type:
        available_models = [model_type]
    
    for ticker_symbol in feature_df[CATEGORY_COL].unique():
        ticker_df_feat = feature_df[feature_df[CATEGORY_COL] == ticker_symbol].drop(CATEGORY_COL, axis=1).copy()
        X_train, y_train = split_feat_df_Xy(ticker_df_feat)

        for model_type in available_models:
            logger.debug(f"Training model [{model_type}] for ticker [{ticker_symbol}]...")
            xgb_model = train_model(X_train, y_train, model_type, ticker_symbol, tune_params, save_model)

            # learning_curves_fig , feat_importance_fig = extract_learning_curves(xgboost_model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Tree-based models with optional hyperparameter tuning.")
    parser.add_argument(
        "-t", "--tune",
        action="store_true",
        help="Enable hyperparameter tuning using GridSearchCV. Defaults to False."
    )
    parser.add_argument(
        "-mt", "--model_type",
        type=str,
        choices=["xgb", "et"],
        help="Model name to train (xgb, et) (optional, defaults to all)."
    )
    parser.add_argument(
        "-ts", "--ticker_symbol",
        type=str,
        help="""Ticker Symbol to train on. (optional, defaults to all).
        Example: bova11 -> BOVA11.SA | petr4 -> PETR4.SA"""
    )
    parser.add_argument(
        "-sm", "--save_model",
        action="store_false",
        help="Disable saving model to file system. Defaults to True. Run '--save_model' to Disable."
    )
    args = parser.parse_args()

    logger.info("Starting the training pipeline...")
    training_pipeline(args.tune, args.model_type, args.ticker_symbol, args.save_model)
    logger.info("Training Pipeline completed successfully!")

# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')

from src.models.evaluate_model import *


def daily_model_evaluation(model_type=None, ticker=None):
    """Performs the models' performance daily evaluation"""

    # Loads the out of sample forecast table and the training dataset
    current_train_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_df.csv'), parse_dates=["DATE"])
    historical_forecasts_df = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, 'forecast_output_df.csv'), parse_dates=["DATE","RUN_DATE"])

    available_models = model_config['available_models']

    latest_price_date = current_train_df["DATE"].max().date()
    latest_run_date = historical_forecasts_df["RUN_DATE"].max().date()

    logger.debug(f"Latest availabe date: {latest_price_date}")
    logger.debug(f"Latest run date: {latest_run_date}")

    # Check the ticker parameter
    if ticker:
        ticker = ticker.upper() + '.SA'
        historical_forecasts_df = historical_forecasts_df[historical_forecasts_df[CATEGORY_COL] == ticker]
        current_train_df = current_train_df[current_train_df[CATEGORY_COL] == ticker]
    
    # Check the model_type parameter 
    if model_type is not None and model_type not in available_models:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from: {available_models}")
    
    elif model_type:
        available_models = [model_type]

    for ticker in historical_forecasts_df[CATEGORY_COL].unique():
        ticker_hist_forecast = historical_forecasts_df[historical_forecasts_df[CATEGORY_COL] == ticker]
        ticker_train_df = current_train_df[current_train_df[CATEGORY_COL] == ticker]

        for model_type in available_models:
            model_type = model_type.upper()

            latest_value = ticker_train_df[TARGET_COL].values[-1]

            try:
                predicted_value = ticker_hist_forecast[
                    (ticker_hist_forecast["MODEL_TYPE"] == model_type) \
                    & (ticker_hist_forecast["RUN_DATE"] == pd.to_datetime(latest_run_date)) \
                    & (ticker_hist_forecast["DATE"] == pd.to_datetime(latest_price_date))
                ][PREDICTED_COL].values[0]

            except:
                logger.error(f"\nThe last forecasts where made at {latest_run_date}, A.K.A today. Comeback tomorrow in order to calculate today's performance.")
                raise ValueError(f"\nThe last forecasts where made at {latest_run_date}, A.K.A today. Comeback tomorrow in order to calculate today's performance.")

            evaluate_and_store_performance(model_type, ticker, latest_value, predicted_value, latest_price_date, latest_run_date)


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
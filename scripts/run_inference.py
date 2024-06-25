# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')

from src.models.predict_model import *


def inference_pipeline(models_list: list[str], ticker_list: list[str], write_to_table=True):
    """
    Generates predictions using a pre-trained model for specified tickers.

    This function iterates through a list of stock tickers and applies a loaded production model to forecast future prices.
    It constructs a future dataframe for each ticker based on a defined forecast horizon,
    performs iterative predictions, and consolidates the results. Optionally, it appends the predictions to a CSV file.

    Args:
        models_list (list): List of available models (currently unused in the function).
        ticker_list (list): List of ticker symbols for stocks to predict.
        write_to_table (bool, optional): If True, appends predictions to a CSV file. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing predictions for each ticker, including columns for DATE, PREDICTION, MODEL_TYPE, and RUN_DATE.

    Process:
        1. Loads featurized dataset.
        2. Filters dataset by each ticker in ticker_list.
        3. Loads the production model for the current ticker.
        4. Creates a future DataFrame for the forecast horizon.
        5. Performs iterative predictions using the loaded model.
        6. Appends predictions with model type and run date.
        7. Concatenates all predictions.
        8. (Optional) Appends predictions to a CSV file.
    """
    logger.debug("Loading the featurized dataset...")

    feature_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_NAME), parse_dates=["DATE"])
    final_predictions_df = pd.DataFrame()

    for ticker in ticker_list:
        filtered_feature_df = feature_df[feature_df[CATEGORY_COL] == ticker].copy()

        # for model_type in models_list:
        logger.debug(f"Performing inferece for ticker [{ticker}]...")

        current_prod_model = load_production_model_sklearn(ticker)
        model_name = type(current_prod_model).__name__

        logger.debug("Creating the future dataframe...")
        future_df = make_future_df(FORECAST_HORIZON, filtered_feature_df, features_list)

        logger.debug("Predicting iteratively...")
        predictions_df = make_iterative_predictions(
            model=current_prod_model,
            future_df=future_df,
            past_target_values=list(filtered_feature_df[TARGET_COL].values)
        )
        predictions_df['MODEL_TYPE'] = model_name
        final_predictions_df = pd.concat([final_predictions_df, predictions_df], axis=0)

    # Add the run date
    RUN_DATE = dt.datetime.today().date()
    final_predictions_df["RUN_DATE"] = RUN_DATE

    if write_to_table:
        logger.info("Writing the predictions to database...")

        file_path = f"{OUTPUT_DATA_PATH}/{OUTPUT_DATA_NAME}"
        if os.path.isfile(file_path):
            final_predictions_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            final_predictions_df.to_csv(file_path, index=False)

        logger.info("Predictions written successfully!")

    return final_predictions_df


if __name__ == "__main__":

    logger.info("Starting the Inference pipeline...")
    inference_pipeline(
        models_list = model_config["available_models"],
        ticker_list = data_config["ticker_list"],
    )
    logger.info("Inference Pipeline completed successfully!")
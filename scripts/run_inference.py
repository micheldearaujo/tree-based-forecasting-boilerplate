# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')

from src.models.predict_model import *


def inference_pipeline(models_list: list, ticker_list: list, write_to_table=True):
    """
    Executes the model inference pipeline to generate stock price predictions.

    This function performs the following steps:

    1. Loads the featurized dataset from 'processed_stock_prices.csv'.
    2. Filters the dataset by ticker symbol if provided.
    3. Filters the models to be used for inference based on the provided model type.
    4. For each selected ticker symbol and model type:
        a. Loads the corresponding production model.
        b. Creates a future DataFrame for the forecast horizon.
        c. Generates predictions using the loaded model.
    5. Concatenates all predictions into a single DataFrame.
    6. Optionally writes the predictions to a file or database table.

    Args:
        model_type (str, optional): The type of model to use for inference. If None, all available models in the configuration will be used. Defaults to None.
        ticker (str, optional): The ticker symbol of the stock to predict. If None, predictions are made for all stocks in the dataset. Defaults to None.
        write_to_table (bool, optional): If True, the predictions will be written to a file or database table. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions for all selected ticker symbols and model types.

    Raises:
        ValueError: If an invalid model type is provided.
    """
    logger.debug("Loading the featurized dataset...")

    feature_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_NAME), parse_dates=["DATE"])
    final_predictions_df = pd.DataFrame()

    for ticker in ticker_list:#feature_df[CATEGORY_COL].unique():
        filtered_feature_df = feature_df[feature_df[CATEGORY_COL] == ticker].copy()

        for model_type in models_list:
            logger.debug(f"Performing inferece for ticker [{ticker}] using model [{model_type}]...")
 
            current_prod_model = load_production_model_sklearn(model_type, ticker)

            logger.debug("Creating the future dataframe...")
            future_df = make_future_df(FORECAST_HORIZON, filtered_feature_df, features_list)

            logger.debug("Predicting...")
            predictions_df = make_iterative_predictions(
                model=current_prod_model,
                future_df=future_df,
                past_target_values=list(filtered_feature_df[TARGET_COL].values)
            )
            predictions_df['MODEL_TYPE'] = model_type
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
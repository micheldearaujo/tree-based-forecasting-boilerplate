# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')


from src.models.train_model import *
from src.utils import weekend_adj_forecast_horizon


def hyperparam_tuning_pipeline(models_list: list[str], ticker_list: list[str], forecast_horizon: int, save_params: bool=True):

    logger.debug("Loading the featurized dataset..")
    feature_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_NAME), parse_dates=["DATE"])
    logger.info(f"Last Available Date in Training Dataset: {feature_df['DATE'].max()}")

    forecast_horizon = weekend_adj_forecast_horizon(forecast_horizon, 2)
    
    for ticker_symbol in ticker_list:
        ticker_df_feat = feature_df[feature_df[CATEGORY_COL] == ticker_symbol].drop(columns=[CATEGORY_COL]).copy()

        # Remove last "forecast_horizon" Period - it will be the hold-out test set for model selection
        hold_out_date = ticker_df_feat["DATE"].max() - relativedelta(days=forecast_horizon)
        ticker_df_feat = ticker_df_feat[ticker_df_feat["DATE"] < hold_out_date].copy()
        X_train, y_train = split_feat_df_Xy(ticker_df_feat)

        logger.info(f"Last Available Date for Tuning: {ticker_df_feat['DATE'].max()}")

        for model_type in models_list:
            logger.debug(f"Performing hyperparameter tuning for ticker [{ticker_symbol}] using {model_type}...")

            grid_search = tune_params_gridsearch(X_train, y_train, model_type, n_splits=3)
            best_params = grid_search.best_params_

            logger.info(f"Best parameters found: {best_params}")
            if save_params:
                os.makedirs(MODELS_PATH, exist_ok=True)
                os.makedirs(os.path.join(MODELS_PATH, ticker_symbol), exist_ok=True)
                joblib.dump(best_params, os.path.join(MODELS_PATH, ticker_symbol, f"best_params_{model_type}.joblib"))

                pd.DataFrame(grid_search.cv_results_) \
                    .to_csv(os.path.join(MODELS_PATH, ticker_symbol, f"cv_results_{model_type}.csv"), index=False)

if __name__ == "__main__":

    logger.info("Starting the Hyperparameter Tuning pipeline...")
    hyperparam_tuning_pipeline(
        models_list = model_config["available_models"],
        ticker_list = data_config["ticker_list"],
        forecast_horizon = FORECAST_HORIZON,
        save_params = True
    )
    logger.info("Hyperparameter Tuning completed successfully!")

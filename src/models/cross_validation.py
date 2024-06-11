# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *


# make the dataset
PERIOD = '800d'
INTERVAL = '1d'

# Execute the whole pipeline

if __name__ == "__main__":

    #STOCK_NAME = str(input("Which stock do you want to track? "))
    STOCK_NAME = 'BOVA11.SA'
    logger.info("Starting the Cross Validation pipeline..")

    # download the dataset
    stock_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)

    # perform featurization
    stock_df_feat = build_features(stock_df, features_list)

    # perform cross-validation until the last forecast horizon
    cross_val_data = stock_df_feat[stock_df_feat.Date < stock_df_feat.Date.max() - dt.timedelta(days=model_config["FORECAST_HORIZON"])]

    best_model, best_params = time_series_grid_search_xgb(
        X=cross_val_data.drop([model_config["TARGET_NAME"], "Date"], axis=1),
        y=cross_val_data[model_config["TARGET_NAME"]],
        param_grid=param_grid,
        n_splits=3,
        random_state=42,
        stock_name=STOCK_NAME
    )

    logger.info("Cross Validation Pipeline was sucessful!")
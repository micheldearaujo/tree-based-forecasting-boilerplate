# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')

from src.features.feat_eng import *


if __name__ == '__main__':

    dummy_columns = ["DAY_OF_MONTH", "DAY_OF_WEEK", "WEEK_OF_MONTH", "MONTH", "QUARTER", "YEAR"]

    logger.debug("Loading the raw dataset to featurize it...")
    raw_df = pd.read_csv(os.path.join(RAW_DATA_PATH, RAW_DATA_NAME), parse_dates=["DATE"])

    logger.info("Featurizing the dataset...")
    features_list = config['features_list']
    feature_df = build_features(raw_df, features_list)
    
    write_dataset_to_file(feature_df, PROCESSED_DATA_PATH, PROCESSED_DATA_NAME)

    logger.debug("Features built successfully!")
    logger.info(f"\n{feature_df.tail()}")
    logger.debug(f"Dataset shape: {feature_df.shape}.")
    logger.debug(f"Amount of ticker symbols: {feature_df[CATEGORY_COL].nunique()}.")
    logger.info("Finished featurizing the dataset!")

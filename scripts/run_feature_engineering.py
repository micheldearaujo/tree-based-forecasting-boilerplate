# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')

from src.features.feat_eng import *

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)

with open("src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())
    data_config = config['data_config']
    model_config = config['model_config']
    RAW_DATA_PATH = data_config['paths']['raw_data_path']
    RAW_DATA_NAME = data_config['table_names']['raw_table_name']
    PROCESSED_DATA_PATH = data_config['paths']['processed_data_path']
    PROCESSED_DATA_NAME = data_config['table_names']['processed_table_name']
    TARGET_COL = model_config['target_col']
    CATEGORY_COL = model_config['category_col']

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
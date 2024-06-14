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

RAW_DATA_PATH = config['paths']['raw_data_path']
PROCESSED_DATA_PATH = config['paths']['processed_data_path']

logger.debug("Loading the raw dataset to featurize it...")
raw_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'raw_df.csv'), parse_dates=["DATE"])

logger.info("Featurizing the dataset...")
features_list = config['features_list']
feat_df = build_features(raw_df, features_list)

write_dataset_to_file(feat_df, PROCESSED_DATA_PATH, "processed_df")

logger.debug("Features built successfully!")
logger.debug(f"\n{feat_df.tail()}")
logger.debug(f"Dataset shape: {feat_df.shape}.")
logger.debug(f"Amount of ticker symbols: {feat_df['STOCK'].nunique()}.")
logger.info("Finished featurizing the dataset!")
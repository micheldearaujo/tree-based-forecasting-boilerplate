# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')

from src.data.make_dataset import *

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)

with open("src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())

logger.info("Downloading the raw dataset...")
stock_df = make_dataset(ticker_list, PERIOD, INTERVAL)
logger.info("Finished downloading the raw dataset!")
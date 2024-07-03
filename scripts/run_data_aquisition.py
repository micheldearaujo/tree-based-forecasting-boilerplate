# -*- coding: utf-8 -*-
import sys
import os
import logging.config
sys.path.insert(0,'.')

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

from src.data.make_dataset import *

if __name__ == "__main__":
    
    logger.info("Downloading the raw dataset...")
    stock_df = make_dataset(ticker_list, PERIOD, INTERVAL)
    logger.info("Finished downloading the raw dataset!")
    logger.info(f"\n{stock_df.tail()}")
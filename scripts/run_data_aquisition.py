# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')

from src.data.make_dataset import *

if __name__ == "__main__":
    
    logger.info("Downloading the raw dataset...")
    stock_df = make_dataset(ticker_list, PERIOD, INTERVAL)
    logger.info("Finished downloading the raw dataset!")
    logger.debug(stock_df.tail())
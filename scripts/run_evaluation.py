# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')

from src.models.evaluate_model import *

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    

with open("src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())
    PROCESSED_DATA_PATH = config['paths']['processed_data_path']
    OUTPUT_DATA_PATH = config['paths']['output_data_path']


parser = argparse.ArgumentParser(description="Perform Out-of-Sample Tree-based models Inference.")
parser.add_argument(
    "-mt", "--model_type",
    type=str,
    choices=["xgb", "et"],
    help="Model name use for inference (xgb, et) (optional, defaults to all)."
)
parser.add_argument(
    "-ts", "--ticker_symbol",
    type=str,
    help="""Ticker Symbol for inference. (optional, defaults to all).
    Example: bova11 -> BOVA11.SA | petr4 -> PETR4.SA"""
)
args = parser.parse_args()

logger.info("Starting the Daily Model Evaluation pipeline...")
daily_model_evaluation(args.model_type, args.ticker_symbol)
logger.info("Daily Model Evaluation Pipeline completed successfully!")
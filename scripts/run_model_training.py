# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')

from src.models.train_model import *

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)

with open("src/configuration/hyperparams.yaml", 'r') as f:  
    hyperparams = yaml.safe_load(f.read())

with open("src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())
    model_config = config['model_config']


parser = argparse.ArgumentParser(description="Train Tree-based models with optional hyperparameter tuning.")
parser.add_argument(
    "-t", "--tune",
    action="store_true",
    help="Enable hyperparameter tuning using GridSearchCV. Defaults to False."
)
parser.add_argument(
    "-mt", "--model_type",
    type=str,
    choices=["xgb", "et"],
    help="Model name to train (xgb, et) (optional, defaults to all)."
)
parser.add_argument(
    "-ts", "--ticker_symbol",
    type=str,
    help="""Ticker Symbol to train on. (optional, defaults to all).
    Example: bova11 -> BOVA11.SA | petr4 -> PETR4.SA"""
)
parser.add_argument(
    "-sm", "--save_model",
    action="store_false",
    help="Disable saving model to file system. Defaults to True. Run '--save_model' to Disable."
)
args = parser.parse_args()

logger.info("Starting the training pipeline...")
training_pipeline(args.tune, args.model_type, args.ticker_symbol, args.save_model)
logger.info("Training Pipeline completed successfully!")

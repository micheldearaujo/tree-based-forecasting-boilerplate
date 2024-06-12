# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'.')

from src.models.predict_model import *

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)

with open("src/configuration/project_config.yaml", 'r') as f:  
    config = yaml.safe_load(f.read())


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
parser.add_argument(
    "-w", "--write_to_table",
    action="store_false",
    help="Disable Writing the OFS forecasts to table. Defaults to True. Run '--write_to_table' to Disable."
)
args = parser.parse_args()

logger.info("Starting the Inference pipeline...")

try:
    model_type =args.model_type.upper()
    inference_pipeline(model_type, args.ticker_symbol, args.write_to_table)
except:
    inference_pipeline(args.model_type, args.ticker_symbol, args.write_to_table)
logger.info("Inference Pipeline completed successfully!")
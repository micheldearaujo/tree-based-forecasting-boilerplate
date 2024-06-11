import sys
sys.path.insert(0,'.')

from src.utils import *
from src.features.feat_eng import build_features
from src.data.make_dataset import make_dataset
from src.models.train_model import train_pipeline
from src.models.predict_model import inference_pipeline
from src.models.hyperparam_tune import hyperopt_tune_pipeline
from src.models.evaluate_model import model_testing_pipeline
from src.models.app import front_end

def execute_full_pipeline():
    """
    Main function that executes the full pipeline.
    
    Parameters:
        None
    Returns:
        None
    """

    #task = asyncio.create_task(front_end())
    logger.info("Starting the full pipeline..\n")
    
    # execute the data ingestion pipeline
    logger.info("Executing the data ingestion pipeline..\n")
    stock_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)
    
    # execute the feature engineering pipeline
    logger.info("Executing the feature engineering pipeline..\n")
    stock_df_feat = build_features(stock_df, features_list)

    # # execute the model Cross val pipeline
    # logger.info("Executing the model tuning pipeline..\n")
    # hyperopt_tune_pipeline()
    
    # execute the model training pipeline
    logger.info("Executing the model training pipeline..\n")
    train_pipeline()

    # execute the model testing pipeline
    logger.info("Executing the model training pipeline..\n")
    model_testing_pipeline()
    
    # execute the model inference pipeline
    logger.info("Executing the model inference pipeline..\n")
    inference_pipeline()
    
    logger.info("Full pipeline executed sucessfully!")




execute_full_pipeline()
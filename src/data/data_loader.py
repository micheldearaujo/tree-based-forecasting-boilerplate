# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import datetime as dt
import pandas as pd

from src.configuration.config_feature import variables_list
from src.features.feat_eng import fill_na_interpolation
from src.configuration.config_model import model_config

TARGET_COL = model_config["target_col"]


def load_and_preprocess_model_dataset(dataset_name: str) -> pd.DataFrame:

    """
    This function loads and preprocesses a dataset for a predictive model.
    It first loads the dataset from a specified source (currently CSV file),
    filters the columns, fills missing values using linear interpolation,
    and then propagates the first and last values to fill any remaining missing values.
    Finally, it sorts the DataFrame by the 'date' column.

    Args:
        dataset_name (str): The name of the dataset to load and preprocess.

    Returns:
        pd.DataFrame: The preprocessed dataset ready for model training or forecasting.
    """

    feature_df = pd.read_csv(f"./data/processed/{dataset_name}.csv", parse_dates=["date"])

    return feature_df


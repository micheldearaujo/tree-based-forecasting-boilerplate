# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

import os
import logging.config
import yaml

import pandas as pd

from src.utils import write_dataset_to_file


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

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)


def create_lag_features(df: pd.DataFrame, lag_values: list, target_column: str = TARGET_COL) -> pd.DataFrame:
    """
    Creates lag features for the specified target column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target column.
        lag_values (list): A list of integers specifying the lag values (e.g., [1, 2, 5] for 1-day, 2-day, and 5-day lags).
        target_column (str, optional): The name of the column to create lag features for (default: TARGET_COL).

    Returns:
        pd.DataFrame: The input DataFrame with additional lag features.
    """

    for lag in lag_values:
        df[f"{target_column}_LAG_{lag}"] = df[target_column].shift(lag).round(3)
    return df


def create_moving_average_features(df: pd.DataFrame, ma_values: list, target_column: str = TARGET_COL) -> pd.DataFrame:
    """
    Creates moving average features for the specified target column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target column.
        ma_values (list): A list of integers specifying the window sizes for the moving averages (e.g., [5, 10] for 5-day and 10-day moving averages).
        target_column (str, optional): The name of the column to create moving average features for (default: TARGET_COL).

    Returns:
        pd.DataFrame: The input DataFrame with additional moving average features.
    """

    for ma in ma_values:
        df[f"{target_column}_MA_{ma}"] = df[target_column].rolling(ma, closed='left').mean().round(3)
    return df


def create_date_features(df: pd.DataFrame, date_column: str = "DATE") -> pd.DataFrame:
    """Creates date-based features from the specified date column."""

    df['DAY_OF_MONTH'] = df[date_column].dt.day
    df['MONTH'] = df[date_column].dt.month
    df['QUARTER'] = df[date_column].dt.quarter
    df['DAY_OF_WEEK'] = df[date_column].dt.weekday
    df['WEEK_OF_MONTH'] = (df['DAY_OF_MONTH'] - 1) // 7 + 1
    df['YEAR'] = df[date_column].dt.year

    return df


def build_features(raw_df: pd.DataFrame, features_list: list, save: bool = True) -> pd.DataFrame:
    """
    Creates features for a machine learning dataset from raw data.

    Args:
        raw_df: Raw Pandas DataFrame containing DATE and TARGET columns, as well as a categorical
        column to split between different series (or objects).
        features_list: List of feature names to create.
        save: Whether to save the processed data to a CSV file (default: True).

    Returns:
        Pandas DataFrame with the new features.
    """
    logger.debug("Building features...")
    feature_df = pd.DataFrame()

    for ticker in raw_df[CATEGORY_COL].unique():
        logger.debug(f"Building features for ticker [{ticker}]...")
        raw_df_filtered = raw_df[raw_df[CATEGORY_COL] == ticker].copy()
        
        raw_df_filtered = create_date_features(df=raw_df_filtered)

        if TARGET_COL in raw_df_filtered.columns:
            raw_df_filtered[TARGET_COL] = raw_df_filtered[TARGET_COL].round(3)

        raw_df_filtered = create_lag_features(raw_df_filtered, lag_values=[int(f.split("_")[-1]) for f in features_list if "LAG" in f])
        raw_df_filtered = create_moving_average_features(raw_df_filtered, ma_values=[int(f.split("_")[-1]) for f in features_list if "MA" in f])
        raw_df_filtered.dropna(inplace=True)

        feature_df = pd.concat([feature_df, raw_df_filtered], axis=0)

    feature_df.columns = feature_df.columns.str.upper() 
    feature_df = feature_df.reindex(columns=["DATE", CATEGORY_COL, TARGET_COL, *features_list])

    # feature_df =  dummy_date_columns(feature_df, dummy_columns)

    return feature_df


def dummy_date_columns(feature_df, dummy_columns):

    for column in dummy_columns:
        feature_df[column] = feature_df[column].astype('str')

    feature_df_dummy = pd.get_dummies(feature_df, columns=dummy_columns, dtype='int')
    feature_df_dummy

    return feature_df_dummy


if __name__ == '__main__':

    dummy_columns = ["DAY_OF_MONTH", "DAY_OF_WEEK", "WEEK_OF_MONTH", "MONTH", "QUARTER", "YEAR"]

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



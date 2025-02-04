# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

import os
import logging.config
import yaml

import pandas as pd
import numpy as np
from scipy.signal import detrend
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

from src.configuration.config_data import *
from src.configuration.config_model import *
from src.configuration.config_feature import *
from src.configuration.config_viz import *

TARGET_COL = model_config['target_col']
PREDICTED_COL = model_config['predicted_col']

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("git.cmd").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

def resample_and_interpolate_monthly_to_weekly(df: pd.DataFrame, original_date_col: str, final_date_col: str):
    """
    Resamples a monthly DataFrame to a weekly frequency and interpolates missing values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing monthly data.
    - original_date_col (str): The name of the column in the DataFrame that contains the original dates.
    - final_date_col (str): The name of the new column to store the resampled dates.

    Returns:
    - pd.DataFrame: The input DataFrame with the original dates resampled to weekly frequency and missing values interpolated.
    """

    df[original_date_col] = df[original_date_col].dt.tz_localize(None)
    df['original_date'] = df[original_date_col].copy()
    df_w = df.set_index(original_date_col).resample('W-FRI').mean().interpolate(method='linear', direction='forward').reset_index()
    df_w = df_w.rename(columns={original_date_col: final_date_col})
    df_w = df_w.sort_values(final_date_col)

    return df_w

def resample_and_interpolate_monthly_to_monthly(df: pd.DataFrame, original_date_col: str, final_date_col:str):
    """
    Resamples a monthly DataFrame to a weekly frequency and interpolates missing values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing monthly data.
    - original_date_col (str): The name of the column in the DataFrame that contains the original dates.
    - final_date_col (str): The name of the new column to store the resampled dates.

    Returns:
    - pd.DataFrame: The input DataFrame with the original dates resampled to weekly frequency and missing values interpolated.
    """

    df[original_date_col] = df[original_date_col].dt.tz_localize(None)
    df['original_date'] = df[original_date_col].copy()
    df_m = df.set_index(original_date_col).resample('ME').asfreq().interpolate(method='linear', direction='forward').reset_index()
    df_m = df_m.rename(columns={original_date_col: final_date_col})
    df_m.drop(columns='original_date', inplace=True)
    df_m = df_m.sort_values(final_date_col)
    return df_m


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
    logger.debug(f'Lag Values for {target_column}: {lag_values}')
    for lag in lag_values:
        df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag).bfill()

    return df


def create_moving_average_features(df: pd.DataFrame, sma_windows: list[int], target_column: str = TARGET_COL) -> pd.DataFrame:
    """
    Creates moving average features for the specified target column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target column.
        sma_windows (list): A list of integers specifying the window sizes for the moving averages
                            (e.g., [5, 10] for 5-periods and 10-periods moving averages).
        target_column (str, optional): The name of the column to create moving average features for (default: TARGET_COL).

    Returns:
        pd.DataFrame: The input DataFrame with additional moving average features.
    """
    logger.debug(f'SMA Values for {target_column}: {sma_windows}')
    for sma_window in sma_windows:
        df[f"{target_column}_sma_{sma_window}"] = df[target_column].rolling(sma_window, closed='left').mean().bfill()

    return df


def create_moving_max_features(df: pd.DataFrame, mm_windows: list[int], target_column: str = TARGET_COL) -> pd.DataFrame:
    """
    Moving max features for the specified column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target column.
        mm_windows (list): A list of integers specifying the window sizes for the moving max (e.g., [5, 10] for 5-periods and 10-periods moving max).
        target_column (str, optional): The name of the column to create moving average features for (default: TARGET_COL).

    Returns:
        pd.DataFrame: The input DataFrame with additional moving max features.
    """
    logger.debug(f'Moving Max Values for {target_column}: {mm_windows}')
    for mm_window in mm_windows:
        df[f"{target_column}_mm_{mm_window}"] = df[target_column].rolling(mm_window, closed='left').max().bfill()

    return df


def create_moving_sum_features(df: pd.DataFrame, ms_windows: list[int], target_column: str = TARGET_COL) -> pd.DataFrame:
    """
    Moving sum features for the specified column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target column.
        ms_windows (list): A list of integers specifying the window sizes for the moving sum (e.g., [5, 10] for 5-periods and 10-periods moving sum).
        target_column (str, optional): The name of the column to create moving average features for (default: TARGET_COL).

    Returns:
        pd.DataFrame: The input DataFrame with additional moving sum features.
    """
    logger.debug(f'Moving Sum Values for {target_column}: {ms_windows}')
    for ms_window in ms_windows:
        df[f"{target_column}_ms_{ms_window}"] = df[target_column].rolling(ms_window, closed='left').sum().bfill()

    return df


def compare_current_value_to_ma(df: pd.DataFrame, ma_values: list, target_column: str):
    """
    Creates a feature that compares the current value of a column with its moving average.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target column.
        ma_values (list): A list of integers specifying the window sizes for the moving averages (e.g., [5, 10] for 5-day and 10-day moving averages).
        target_column (str, optional): The name of the column to create comparison moving average features for (default: TARGET_COL).

    Returns:
        pd.DataFrame: The input DataFrame with additional moving average features.
    """
    logger.debug(f'Comparison SMA Values for {target_column}: {ma_values}')
    for ma in ma_values:
        print(ma)
        df[f"{target_column}_sma_{ma}"] = df[target_column].rolling(ma, closed='left').mean().bfill().replace(0, pd.NA)
        df[f"comparison_{target_column}_sma_{ma}"] = (df[target_column] / df[f"{target_column}_sma_{ma}"])
        df = df.drop(columns=[f"{target_column}_sma_{ma}"])
    
    return df


def compare_shift_values(df: pd.DataFrame, lags: list, target_column: str) -> pd.DataFrame:
    """
    Creates comparison lag features for the specified target column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target column.
        lags(list): A list of integers specifying the lags sizes to compare
        target_column (str, optional): The name of the column to create the lag comparison for (default: TARGET_COL).

    Returns:
        pd.DataFrame: The input DataFrame with additional lag comparison features.
    """
    df[f"comparison_{target_column}_shift_{lags[0]}_{lags[1]}"] = 100 * (df[target_column].shift(lags[0]).bfill() / df[target_column].shift(lags[1]).bfill().replace(0, pd.NA))
    df[f"comparison_{target_column}_shift_{lags[0]}_{lags[1]}"].bfill().ffill()

    return df


def compare_moving_sum_values(df: pd.DataFrame, windows: list, target_column: str) -> pd.DataFrame:
    """
    Creates comparison moving sum features for the specified target column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target column.
        lags(list): A list of integers specifying the lags sizes to compare
        target_column (str, optional): The name of the column to create the lag comparison for (default: TARGET_COL).

    Returns:
        pd.DataFrame: The input DataFrame with additional lag comparison features.
    """

    df[f"comparison_{target_column}_ms_{windows[0]}_{windows[1]}"] = \
        100 * \
                ((
            df[target_column].rolling(windows[0], closed="left", min_periods=3).sum().bfill() /
            df[target_column].rolling(windows[1], closed="left", min_periods=3).sum().bfill().replace(0, pd.NA)
        ) \
        .bfill().ffill()) \

    return df


def compare_moving_average_values(df: pd.DataFrame, windows: list, target_column: str) -> pd.DataFrame:
    """
    Creates comparison moving sum features for the specified target column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target column.
        lags(list): A list of integers specifying the lags sizes to compare
        target_column (str, optional): The name of the column to create the lag comparison for (default: TARGET_COL).

    Returns:
        pd.DataFrame: The input DataFrame with additional lag comparison features.
    """

    df[f"comparison_{target_column}_sma_{windows[0]}_{windows[1]}"] = \
        100 *(
                (df[target_column].rolling(windows[0], closed="left", min_periods=3).mean().bfill() /
                df[target_column].rolling(windows[1], closed="left", min_periods=3).mean().bfill().replace(0, pd.NA)
            ).bfill().ffill()
        )

    return df


def create_date_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Creates date-based features from the specified date column."""

    # df['day_of_month'] = df[date_column].dt.day
    df['month'] = df[date_column].dt.month
    # df['quarter'] = df[date_column].dt.quarter
    # df['day_of_week'] = df[date_column].dt.weekday
    # df['week_of_month'] = (df['DAY_OF_MONTH'] - 1) // 7 + 1
    # df['year'] = df[date_column].dt.year
    # df.drop(columns=['day_of_month', 'day_of_week'], inplace=True)

    return df


def fill_na_interpolation(df: pd.DataFrame, variables_to_fill: list) -> pd.DataFrame:
    """
    Performs linear interpolation on specified variables in a DataFrame to fill missing values,
    assuming a weekly frequency.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the variables to interpolate.
        variables_to_interpolate (list): A list of column names in the DataFrame to interpolate.

    Returns:
        pd.DataFrame: The input DataFrame with new columns containing the interpolated values.
    """

    for variable in variables_to_fill:
        df[variable] = df[variable].interpolate(method='linear', limit_direction='forward')

    return df


def smooth_moody_index(df: pd.DataFrame, original_moody_col: str, new_moody_col: str, n_lags: int) -> pd.DataFrame:
    """
    This function smooths the Moody's index values in a DataFrame by calculating an exponentially weighted moving average (EWMA)
    of the original Moody's index column, shifted by a specified number of lags. Missing values are filled using backward fill (bfill).

    Args:
        df (pd.DataFrame): The input DataFrame containing the original Moody's index column.
        original_moody_col (str): The name of the original Moody's index column in the DataFrame.
        new_moody_col (str): The name of the new column to store the smoothed Moody's index values.
        n_lags (int): The number of lags to shift the original Moody's index column before calculating the EWMA.

    Returns:
        pd.DataFrame: The input DataFrame with the new column 'new_moody_col' containing the smoothed Moody's index values.
    """

    interim_col = f'moody_lagged_{n_lags}'
    df[interim_col] = df[original_moody_col].shift(n_lags)
    df[f'{new_moody_col}_{n_lags}'] = df[interim_col].ewm(alpha=0.109).mean().bfill().ffill()
    df = df.drop(columns=[interim_col])

    return df

def detrend_series(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    This function detrend values in a DataFrame

    Args:
        df (pd.DataFrame): The input DataFrame containing the original Moody's index column.
        target_col (str): The name of the columns to detrend.
    Returns:
        pd.DataFrame: The input DataFrame with the new column detrended.
    """

    df[target_col] = detrend(df[target_col])

    return df


def create_spread_feature(df: pd.DataFrame, lag_to_diff: int, first_col: str, second_col: str, new_col: str) -> pd.DataFrame:
    """
    Created the Spread between two variables.

    Args:
        df (pd.DataFrame): The input DataFrame containing the 'target_col' and 'shfe_col' columns.
        lag_to_diff (int): The number of lags to shift the 'target_col' column before subtracting.
        fist_col (str): The name of the column in the DataFrame to be subtracted from.
        second_col (str): The name of the column in the DataFrame to subtract from the lagged 'first_col' column.
        new_col (str): The name of the new column to store the calculated spread values.

    Returns:
        pd.DataFrame: The input DataFrame with the new column 'new_col' containing the calculated spread values.
    """

    df[f"{new_col}_lag_{lag_to_diff}"] = (df[first_col] - df[second_col])
    df[f"{new_col}_lag_{lag_to_diff}"] = df[f"{new_col}_lag_{lag_to_diff}"].shift(lag_to_diff).bfill()

    return df


def calculate_bollinger_bands(df: pd.DataFrame, date_col: str, target_column: str, window_size: int=20, num_std_dev: int=2) -> pd.DataFrame:
    """
    Calculates Bollinger Bands for a target column in a Pandas DataFrame with a datetime index.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the target column and datetime index.
        target_column (str): The name of the column for which to calculate Bollinger Bands.
        window_size (int, optional): The rolling window size for calculating the moving average and standard deviation (default: 20).
        num_std_dev (int, optional): The number of standard deviations to use for the upper and lower bands (default: 2).

    Returns:
        pandas.DataFrame: The original DataFrame with three new columns:
            - 'bband_spread_': The spread between the upper band and the lower band.
    """

    # Check if DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        #raise ValueError("DataFrame must have a datetime index.")
        df = df.set_index(date_col)

    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Calculate rolling mean and standard deviation
    rolling_mean = df[target_column].rolling(window=window_size, closed='left').mean()
    rolling_std = df[target_column].rolling(window=window_size, closed='left').std()

    # Calculate Bollinger Bands
    df[f'middle_bband_{window_size}'] = rolling_mean
    df[f'upper_bband_{window_size}'] = rolling_mean + (rolling_std * num_std_dev)
    df[f'lower_bband_{window_size}'] = rolling_mean - (rolling_std * num_std_dev)
    df[f'bband_spread_{window_size}'] = (df[f'upper_bband_{window_size}'] - df[f'lower_bband_{window_size}']).bfill()
    df = df.reset_index()
    df.drop(columns = [f'middle_bband_{window_size}',f'upper_bband_{window_size}', f'lower_bband_{window_size}'], inplace=True)

    return df


def calculate_rsi(df: pd.DataFrame, target_col: str, window: int=20) -> pd.DataFrame:
    """
    Calculate the Relative Strength Index (RSI) for a given time series.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the time series data.
    - target_col (str): Name of the column containing the time series values.
    - window (int): The number of periods to use for calculating the RSI (default is 14).

    Returns:
    - pd.DataFrame: Original DataFrame with an additional column for RSI.
    """
    # Calculate the daily changes
    df['change'] = df[target_col].diff()

    # Calculate gains and losses
    df['gain'] = df['change'].apply(lambda x: max(x, 0))
    df['loss'] = df['change'].apply(lambda x: abs(min(x, 0)))

    # Calculate the average gains and losses
    df['avg_gain'] = df['gain'].rolling(window=window, min_periods=1, closed = 'left').mean()
    df['avg_loss'] = df['loss'].rolling(window=window, min_periods=1, closed = 'left').mean()

    # Calculate the Relative Strength (RS)
    df['rs'] = df['avg_gain'] / df['avg_loss'].replace(0, pd.NA)

    # Calculate the RSI
    df[f"rsi_{window}"] = (100 - (100 / (1 + df['rs'])))
    df[f"rsi_{window}"] = df[f"rsi_{window}"].apply(lambda x: None if pd.isna(x) else x).bfill()

    # Drop the intermediate columns used for calculation
    df.drop(columns=['change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], inplace=True)

    return df


def add_reset_consecutive_increase_column(df, target_column, reset_periods):

    """
    Add a new column to the DataFrame that counts consecutive increases in a specified column, 
    resetting the count if there are more than a specified number of periods without increases.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame to which the new column will be added.
    column_name : str
        The name of the column in `df` on which to calculate consecutive increases.
    reset_periods : int
        The number of periods without increases required before resetting the count.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with an additional column 'Consecutive_Increases' that contains the count of
        consecutive increases in the specified column, resetting after `reset_periods` periods
        without an increase.

    Example:
    --------
    >>> import pandas as pd
    >>> data = {'Values': [1, 2, 3, 1, 2, 1, 3, 4, 1, 2, 1, 2, 3]}
    >>> df = pd.DataFrame(data)
    >>> df = add_reset_consecutive_increase_column(df, 'Values', 2)
    >>> print(df)
       Values  Consecutive_Increases
    0       1                      0
    1       2                      1
    2       3                      2
    3       1                      0
    4       2                      1
    5       1                      0
    6       3                      1
    7       4                      2
    8       1                      0
    9       2                      1
    10      1                      0
    11      2                      1
    12      3                      2

    Notes:
    -----
    - The function assumes that the `column_name` exists in the DataFrame `df`.
    - The function handles the rolling window of non-increasing periods by using a rolling sum to count
      the periods without increases.
    - If the DataFrame contains missing values or the column specified has non-numeric data, additional
      data cleaning may be required before applying this function.
    """

    
    # Check if the column exists
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' does not exist in the DataFrame")

    # Extract the column values
    values = df[target_column]
    
    # Create a boolean series where True represents an increase
    increases = values > values.shift(1)
    
    # Create a series to count periods without increases
    no_increase_count = (~increases).astype(int).rolling(window=reset_periods + 1, min_periods=1).sum()
    
    # Reset counts if there are more than `reset_periods` periods without increases
    reset_condition = (no_increase_count > reset_periods).astype(int)
    reset_group_id = (reset_condition != reset_condition.shift(1)).cumsum()
    
    # Calculate the count of consecutive increases within each group
    df[f"consecutive_increases_{reset_periods}"] = increases.groupby(reset_group_id).cumsum()
    
    # Replace NaN values (from the shift operation) with 0
    df[f"consecutive_increases_{reset_periods}"] = df[f"consecutive_increases_{reset_periods}"].fillna(0).astype(int)
    
    return df


def replace_infinites_with_null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces infinite values in a DataFrame with NaN values.
    """
    return df.replace([np.inf, -np.inf], np.nan)


def drop_original_columns(feature_df: pd.DataFrame, original_columns_list: list) -> pd.DataFrame:
    """
    This function drops the original columns from a DataFrame based on a given list of column names.
    The only columns that will be kept are the featurized ones.
    """

    columns_to_drop = [col for col in original_columns_list if col in feature_df.columns] 
    df = feature_df.drop(columns=columns_to_drop).copy()

    return df


def build_features(X_train, y_train):
    """
    Prepares a DataFrame for a machine learning model by creating lag features, moving average features, 
    comparing shift values, calculating Bollinger Bands, handling exogenous features, smoothing Moody's 
    Investors Sentiment Index, and dropping unnecessary columns.

    Args:
        X_train (pandas.DataFrame): The DataFrame containing the independent variables.
        y_train (pandas.DataFrame): The DataFrame containing the target variable.

    Returns:
        pandas.DataFrame: The prepared DataFrame for the machine learning model.
    """

    model_df = X_train.join(y_train)
    model_df = model_df.reset_index()

    # model_df = create_date_features(model_df)

    # Target variable features
    model_df = create_lag_features(model_df, lags_target_list)
    model_df = create_moving_average_features(model_df, sma_target_values)
    model_df = create_moving_sum_features(model_df, moving_sum_target_values)

    for col in lags_comparison_target_dict:
        for lags in lags_comparison_target_dict[col]:
            model_df = compare_shift_values(model_df, lags, col)
    
    for col in ms_comparison_target_dict:
        for lags in ms_comparison_target_dict[col]:
            model_df = compare_moving_sum_values(model_df, lags, col)

    for col in sma_comparison_target_dict:
        for lags in sma_comparison_target_dict[col]:
            model_df = compare_moving_average_values(model_df, lags, col)

    for bb_window in bollinger_bands_values:
        model_df = calculate_bollinger_bands(model_df, 'date', TARGET_COL, window_size=bb_window, num_std_dev=2)

    for rsi_window in rsi_windows_list:
        model_df = calculate_rsi(df=model_df, target_col=TARGET_COL, window=rsi_window)
    
    for spread_feature in spread_features_dict:
        model_df = create_spread_feature(
            df = model_df,
            lag_to_diff = 6,
            first_col = spread_features_dict[spread_feature][0],
            second_col = spread_features_dict[spread_feature][1],
            new_col = spread_feature
        )
    
    # Exogenous Features
    for col in lags_exog_dict:
        model_df = create_lag_features(model_df, lags_exog_dict[col], col)
    for col in sma_exog_dict:
        model_df = create_moving_average_features(model_df, sma_exog_dict[col], col)

    for col in lags_comparison_exog_dict:
        for lags in lags_comparison_exog_dict[col]:
            model_df = compare_shift_values(model_df, lags, col)

    for col in sma_comparison_exog_dict:
        model_df = compare_current_value_to_ma(model_df, sma_comparison_exog_dict[col], col)

    model_df = replace_infinites_with_null(model_df)
    model_df = drop_original_columns(model_df, variables_list)
    model_df.drop(columns=[TARGET_COL], inplace=True)
    
    return model_df
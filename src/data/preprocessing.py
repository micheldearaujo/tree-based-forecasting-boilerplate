import sys
sys.path.insert(0,'.')

import yaml
import logging
import numpy as np
import pandas as pd
# import pandas_gbq
from datetime import datetime, timedelta

with open("src/configuration/logging_config.yaml", 'r') as f:  
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)

class Preprocessing:
    """
    How to call this:
        from src.data.preprocessing import Preprocessing
        preprocessor = Preprocessing()
        Preprocessing.run_all()
    """
    
    def __init__(self):
        # self.save_to_table = save_to_table
        self.raw_data_path = "./data/raw/"
        self.interim_data_path = "./data/interim/"
        self.read_project_id = 'google_bq_projet'
        self.write_project_id = 'google_bq_projet'


    def clean_colums_names(self, df):
        """
        This function takes a pandas DataFrame as input and performs the following operations:
        1. Converts all column names to lowercase.
        2. Replaces all spaces in column names with underscores.
        """
        df.columns = df.columns \
            .str.lower() \
            .str.replace(' ', '_') \
            .str.replace('_(', '_') \
            .str.replace(')_', '_') \
            .str.replace('(', '') \
            .str.replace(')', '') \
            .str.replace('/', '') \
            .str.replace('__', '_') \
            .str.replace('$t', 'usdt')

        return df
    

    def excel_date_to_datetime(self, excel_date):
        """
        Converts an Excel date to a datetime object.
        Excel uses 1900 as the base year, with day 1 being 01/01/1900
        Note: Excel incorrectly considers 1900 as a leap year.
        
        Parameters:
            excel_date (int): The Excel date to convert.
        
        Returns:
            datetime.datetime: The datetime object representing the given Excel date.
        """
        base_date = datetime(1899, 12, 30)
        return base_date + timedelta(days=excel_date)
    

    def align_to_weekday_resample(self, df, date_col, desired_weekday):
        """
        Aligns a time series with irregular weekdays to the desired weekday using resampling.
        
        Args:

            df (pandas.DataFrame): The DataFrame with the time series data.
            date_col (str): The name of the column containing datetime values.
            target_col (str): The name of the column containing the time series values.
            desired_weekday (str or int): The desired weekday to align to (e.g., 'MON' for Monday, etc.).
        
        Returns:
            pandas.DataFrame: A new DataFrame with the time series aligned to the desired weekday.
        """

        # Ensure correct data types
        df[date_col] = pd.to_datetime(df[date_col])

        df_aligned = df.copy()

        df_aligned['original_date'] = df_aligned[date_col]
        df_aligned.set_index(date_col, inplace=True)

        # Resample to weekly frequency starting on the desired weekday
        df_aligned = df_aligned.resample(f'W-{desired_weekday}').first()

        df_aligned = df_aligned.reset_index()
        df_aligned[date_col] = pd.to_datetime(df_aligned[date_col])

        return df_aligned


    def get_friday_of_same_week(self, input_date):
        """
        Calculates the Friday of the same week as a given date.
        
        Parameters:
            input_date (datetime.datetime): The date for which to find the Friday.
        
        Returns:
            datetime.datetime: The Friday of the same week as the input date.
        """
        day_of_week = input_date.dayofweek
        days_to_friday = 4 - day_of_week
        friday_date = input_date + pd.Timedelta(days=days_to_friday)
        
        return friday_date


    def closest_first_day(self, date):
        """
        Finds the first day of the month for a given date.
        
        Args:
            date (datetime.datetime): The date for which to find the first day of the month.
        
        Returns:
            datetime.datetime: The first day of the month for the given date.        
        """
        if date.day <= 15:
            return pd.Timestamp(year=date.year, month=date.month, day=1)
        else:
            next_month = date + pd.DateOffset(months=1)
            return pd.Timestamp(year=next_month.year, month=next_month.month, day=1)


    def preprocess_your_data(self, filename, save_to_table=False):
        """
        Performs your data preprocessing here
        """

        df = pd.read_excel(f"{self.raw_data_path}{filename}")
        df = self.clean_colums_names(df)
 
        if save_to_table:
            logger.debug(f'Writing Your Table to Data Lake...')
            # pandas_gbq.to_gbq(dataframe=df, destination_table='artefact.interim_cpi', project_id=self.write_project_id, if_exists='replace')
            df.to_csv(f"{self.interim_data_path}/clean_df.csv", index=False)
        return df

    @classmethod
    def run_all(cls):
        instance = cls()
        save_to_table=True

        ######## Here you will find how to call the preprocessing for each one of the
        ######## Available databases, although the majority of them will be commented
        ######## since only a small portion is frequently used throughout the project.
        ######## To run the Production version of the project, refer to apps/01-preprocessing_and_join.

        logger.info(f"Starting the preprocessing pipeline...")
        clean_cpi = instance.preprocess_your_data(filename='raw_df.xlsx', save_to_table=save_to_table)
  
        logger.info(f"Finished the preprocessing pipeline!")

# Usage
if __name__ == "__main__":
    Preprocessing().run_all()

# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

import yfinance as yfin
import logging.config
import yaml

from src.utils import *

with open("src/configuration/logging_config.yaml", 'r') as f:  

    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)


def fetch_current_stock_price(ticker: str) -> float:
    """
    Fetches the current price for a given stock ticker.

    Parameters:
        ticker (str): The ticker symbol of the stock.

    Returns:
        float: The current price of the stock, or None if the price could not be retrieved.
    """
    try:
        ticker_info = yfin.Ticker(ticker)
        current_price = ticker_info.info.get("currentPrice") 
        return current_price
    except Exception as e:
        logger.error(f"Error fetching current price for {ticker}: {e}")
        return None 
    

def fetch_current_stock_price_df(ticker: str) -> pd.DataFrame:
    """
    Fetches the current stock price and returns it as a DataFrame in the same format as historical data.

    Parameters:
        ticker (str): The stock ticker symbol.

    Returns:
        pandas.DataFrame: A DataFrame containing the current price data with 'Stock', 'Date', and 'Close' columns.
    """
    current_price = fetch_current_stock_price(ticker)  # Reuse your existing function

    if current_price is None:
        return pd.DataFrame()  # Return an empty DataFrame if fetching failed

    today = dt.date.today()  # Get today's date
    data = {
        "Stock": [ticker],
        "Date": [today],
        "Close": [current_price]
    }

    return pd.DataFrame(data)


def fetch_historical_stock_price_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Download data of the closing prices of a given stock and return as Pandas DataFrame.
    
    Parameters:
        ticker (str): The ticker symbol of the stock to retrieve data for.
        period (str): The length of time to retrieve data for, e.g. '1d', '1mo', '3mo', '6mo', '1y', '5y', 'max'.
        interval (str): The frequency of the data, e.g. '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'.
    
    Returns:
        pandas.DataFrame: The dataframe containing the closing price of a single ticker symbol.
    """
    logger.debug(f"Downloading data for Ticker: {ticker}...")

    stock_price_df = yfin.Ticker(ticker).history(period=period, interval=interval)
    
    stock_price_df["Stock"] = ticker
    stock_price_df = stock_price_df[["Stock", "Close"]]
    stock_price_df = stock_price_df.reset_index()

    stock_price_df["Date"] = pd.to_datetime(stock_price_df["Date"])
    stock_price_df["Date"] = stock_price_df["Date"].apply(lambda x: x.date())
    stock_price_df["Date"] = pd.to_datetime(stock_price_df["Date"])

    return stock_price_df


def make_dataset(ticker: str, period: str, interval: str, save_to_table: bool = True) -> pd.DataFrame:
    """
    Creates a dataset of the closing prices of a given stock.
    
    Parameters:
        ticker (str): The name of the stock to retrieve data for.
        period (str): The length of time to retrieve data for, e.g. '1d', '1mo', '3mo', '6mo', '1y', '5y', 'max'.
        interval (str): The frequency of the data, e.g. '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'.
    
    Returns:
        pandas.DataFrame: The dataframe containing the closing prices all stocks.
    """
    raw_df = pd.DataFrame()
    
    for ticker in stocks_list:

        stock_price_df = fetch_historical_stock_price_data(ticker=ticker, period=period, interval=interval)
        # current_price_df = fetch_current_stock_price_df(ticker)

        # stock_price_df = pd.concat([stock_price_df, current_price_df])

        raw_df = pd.concat([raw_df, stock_price_df], axis=0)

    
    raw_df.columns = raw_df.columns.str.upper() 

    # Save the dataset
    if save_to_table:
        raw_df.to_csv(os.path.join(RAW_DATA_PATH, 'raw_stock_prices.csv'), index=False)

    logger.debug(raw_df.tail())

    return raw_df


if __name__ == '__main__':


    logger.info("Downloading the raw dataset...")

    stock_df = make_dataset(stocks_list, PERIOD, INTERVAL)

    logger.info("Finished downloading the raw dataset!")
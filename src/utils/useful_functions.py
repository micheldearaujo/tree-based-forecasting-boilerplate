import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from scipy.stats import shapiro
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from scipy.fft import fft, fftfreq
from scipy.signal import detrend
from scipy import stats


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.configuration.config_data import *
from src.configuration.config_model import *
from src.configuration.config_feature import *
from src.configuration.config_viz import *


# PLOT SERIES ####################################################
def plot_series(df, date_column, var):

    """Generate a series plot graph from the data

    Parameters
    ----------
    df : Dataframe
    date_column : Date column
        Date Column, it should be in the date format.
    var : Variable displayed in the y axis
        Variable, it can be a list or a str

    Returns
    -------
    plotly figure
    """

    colors = ['#002244', '#ff0066', '#65CCCC', '#A349A4', '#DCEBF8']
    if type(var) == str:
        fig = go.Figure()
        #fig.add_trace(go.Scatter(x=df[date_column], y=df[var], mode="markers", name=var, line=dict(color=colors[0])))
        fig.add_trace(go.Scattergl(x=df[date_column], y=df[var], mode="lines+markers", name=var, line=dict(color=colors[0])))
        fig.update_layout({"title":"Série: " + var, "yaxis_title":'', "paper_bgcolor": "#FFFFFF", "plot_bgcolor": "#FAFAFA"})
        fig.show()
    else:
        fig = go.Figure()
        for i, var in enumerate(var):
            # fig.add_trace(go.Scatter(x=df[date_column], y=df[var], mode="lines", name=var, line=dict(color=colors[i]), markers=True))
            fig.add_trace(go.Scattergl(x=df[date_column], y=df[var], mode="lines+markers", name=var, line=dict(color=colors[i]), markers=True))
        fig.update_layout({"title":"Séries", "yaxis_title":'', "paper_bgcolor": "#FFFFFF", "plot_bgcolor": "#FAFAFA"})
        fig.show()


# DETECT OUTLIERS ####################################################

def detect_outliers(df, var, quantiles = [0.01,0.99]):

    upper_limit = df[var].quantile(0.99)
    lower_limit = df[var].quantile(0.01)
    """Flags forecasts that fall outside of uncertainty interval as outliers.

    Parameters
    ----------
    pred : pd.DataFrame
        Dataframe containing forecasts and ground truth on both training and validation data.

    Returns
    -------
    pd.DataFrame
        Prediction dataframe with a new boolean column named "outlier".
    """
    outliers = df.copy()
    outliers['outlier'] = 0
    outliers.loc[outliers[var] > upper_limit, 'outlier'] = 1
    outliers.loc[outliers[var] < lower_limit, 'outlier'] = 1

    outliers['upper_limit'] = upper_limit
    outliers['lower_limit'] = lower_limit
    return outliers

def plot_outliers(outliers_df, var, date_column):
    """Plots percentile bands and detected outliers.

    Parameters
    ----------
    outliers_df : dataframe
        Width of uncertainty interval (for trend uncertainty). Between 0 and 1.
    """
    fig = go.Figure()
    fig.add_trace(
            go.Scatter(
                x=outliers_df.loc[outliers_df['outlier'] == 1][date_column],
                y=outliers_df.loc[outliers_df['outlier'] == 1][var],
                mode="markers",
                name='Outlier',
                marker=dict(color='#002244', size=5),
            )
        )
    fig.add_trace(
            go.Scatter(
                x=outliers_df[date_column],
                y=outliers_df["upper_limit"],
                fill='tonexty',
                mode="lines",
                name='Upper bound',
                line=dict(color='#ff0066', width=0.5),
            )
        )
    fig.add_trace(
            go.Scatter(
                x=outliers_df[date_column],
                y=outliers_df["lower_limit"],
                fill='tonexty',
                mode="lines",
                name='Lower bound',
                line=dict(color='#ff0066', width=0.5),
            )
        )
    fig.update_layout(showlegend=False, 
                      title=f'Outlier detection',
                      title_x=0.5, title_y=0.9)
    fig.show()


# HISTOGRAM ##########################
def histogram(df, var):
    fig = px.histogram(df, x=var, color_discrete_sequence = ['#002244'], opacity=0.75)
    fig.update_layout({"title":"Histograma: " + var, "yaxis_title":'', "paper_bgcolor": "#FFFFFF", "plot_bgcolor": "#FAFAFA"})
    fig.show()

# DUPLICATE ROWS #############################
def duplicate_rows_percentage(df, subset=None):
    """
    Calculate the percentage of duplicate rows in the DataFrame, optionally considering only a subset of columns.
    
    Parameters:
    - df: Pandas DataFrame
    - subset: Optional list of column names to consider for identifying duplicates
    
    Returns:
    - Float representing the percentage of duplicate rows
    """
    # Count duplicate rows
    num_duplicates = df.duplicated(subset=subset).sum()
    
    # Total number of rows in the DataFrame
    total_rows = df.shape[0]
    
    # Calculate percentage of duplicate rows
    percentage_duplicates = (num_duplicates / total_rows) * 100
    
    return percentage_duplicates

# MISSING VALUES #############################
def missing_values_percentage(df):
    """
    Calculate the percentage of missing values in each column of the DataFrame.
    
    Parameters:
    - df: Pandas DataFrame
    
    Returns:
    - Pandas Series containing the percentage of missing values for each column
    """
    # Calculate percentage of missing values in each column
    percentage_missing = df.isnull().mean() * 100
    
    return percentage_missing



# ROLLING MEAN ##########################
def rolling_mean(df, identifier_columns, rolled_column, date_column, start, end):

    if identifier_columns == None:
        df = df.sort_values(date_column)

        # Calculating the rolling metric
        rolled_col = (
            df[rolled_column]
            .rolling(end - start + 1).mean()
        )
    
    else:

        df = df.sort_values(identifier_columns + [date_column]).reset_index(drop=True)

         # Calculating the rolling metric
        rolled_col = (
            df
            .groupby(identifier_columns)[rolled_column]
            .apply(
                lambda group: group.shift(start).rolling(end - start + 1).mean()
            )
        )
    
    return rolled_col


def rolling_std(df, identifier_columns, rolled_column, date_column, start, end):

    if identifier_columns == None:
        df = df.sort_values(date_column)

        # Calculating the rolling metric
        rolled_col = (
            df[rolled_column]
            .rolling(end - start + 1).mean()
        )
    
    else:

        df = df.sort_values(identifier_columns + [date_column]).reset_index(drop=True)

         # Calculating the rolling metric
        rolled_col = (
            df
            .groupby(identifier_columns)[rolled_column]
            .apply(
                lambda group: group.shift(start).rolling(end - start + 1).mean()
            )
        )
    
    return rolled_col

def rolling_quantile(df, identifier_columns, rolled_column, date_column, start, end, quant):

    if identifier_columns == None:
        df = df.sort_values(date_column)

        # Calculating the rolling metric
        rolled_col = (
            df[rolled_column]
            .rolling(end - start + 1).quantile(quant)
        )
    
    else:

        df = df.sort_values(identifier_columns + [date_column]).reset_index(drop=True)

         # Calculating the rolling metric
        rolled_col = (
            df
            .groupby(identifier_columns)[rolled_column]
            .apply(
                lambda group: group.shift(start).rolling(end - start + 1).quantile(quant)
            )
        )
    
    return rolled_col


# PLOT SERIES WITH BANDS #################
def plot_series_band(df, date_col, stat_col, lower_limit_col, upper_limit_col, variable_name, title='Serie'):

    fig = go.Figure()
    # Add the upper and lower bounds as filled areas for the confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([df[date_col], df[date_col][::-1]]),
        y=np.concatenate([df[upper_limit_col], df[lower_limit_col][::-1]]),
        fill='toself',
        fillcolor='#DCEBF8',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Band'
    ))
    # Add the main line plot
    fig.add_trace(go.Scatter(x=df[date_col], y=df['Variable'], mode='lines+markers', name=variable_name, line=dict(color='#002244')))
    fig.add_trace(go.Scatter(x=df[date_col], y=df[stat_col], mode='lines', name=stat_col, line=dict(color='#ff0066')))

    # Update layout
    fig.update_layout({"title":title, "yaxis_title":'', "paper_bgcolor": "#FFFFFF", "plot_bgcolor": "#FAFAFA", 'height': 500, 'width': 1200})

    # Show the plot
    fig.show()



def create_lagged_features(df, target_column, column_to_lag, max_lag):
    """
    Create lagged features for a given column in the dataframe.

    Args:
    df (pd.DataFrame): Input dataframe.
    column (str): Column name to create lags for.
    max_lag (int): Maximum lag to create.

    Returns:
    pd.DataFrame: Dataframe with new lagged columns.
    """
    lags_df = df.copy()
    for lag in range(1, max_lag + 1):
        lags_df[f"{column_to_lag}_lag{lag}"] = lags_df[column_to_lag].shift(lag)

    lags_df = lags_df[[target_column, *[col for col in lags_df.columns if column_to_lag in col]]]
    return lags_df

def calculate_correlations(df, target_column, method='pearson'):
    """
    Calculate the correlation matrix for the target column with all other columns.

    Args:
    df (pd.DataFrame): Input dataframe.
    target_column (str): Target column name.

    Returns:
    pd.DataFrame: Correlation matrix.
    """
    correlation_matrix = df.corr(method=method)[[target_column]]
    return correlation_matrix

def plot_heatmap(correlation_matrix):
    """
    Plot a heatmap of the correlation matrix.

    Args:
    correlation_matrix (pd.DataFrame): Correlation matrix.
    """
    plt.figure(figsize=(4, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()


def test_for_normality(data, column_name, ci=0.05):

    result = shapiro(data[column_name])
    statistic = result.statistic
    p_value = result.pvalue

    print('-'*20, 'Testing for Normality using Shapiro-Wilk test', '-'*20)
    print(f'Statistic: {statistic}')
    print(f'P-value: {p_value}')

    if p_value <= ci:
        print(f'\n!!!!! The |{column_name}| series DOES NOT follow Normal Distribution !!!!!!')
    else:
        print(f'\nThe |{column_name}| series data FOLLOWS a Normal Distribution!')


def stationarity_test(data, column_name):
    # Perform the ADF test
    result = adfuller(data[column_name])
    
    # Extract and print the test results
    print('-'*20, 'Testing for Stationarity using ADF test', '-'*20)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    
    # Interpretation
    if result[1] <= 0.05:
        print(f"\nThe |{column_name}| series IS STATIONARY with 95% confidence level.")
    else:
        print(f"\nThe |{column_name}| series IS NOT stationary.")



# HURST EXPONENT ##########
def hurst_exponent(data):
    """ 
    Returns the Hurst Exponent of the time series vector data. It is calculated using the Rescaled range approach.
    """
    # Create a range of lag values
    lags = range(4, 20)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


# RSI INDEX ##########
def rsi_index(prices, window=14):
    """
    Calculate Relative Strength Index (RSI) for given prices.

    Parameters:
    - prices: A Pandas Series of closing prices.
    - window: The window size for computing gains and losses (default is 14).

    Returns:
    - rsi: Pandas Series containing RSI values.
    """

    # Calculate differences between prices
    deltas = prices.diff()

    # Define gain and loss
    gain = deltas.where(deltas > 0, 0)
    loss = -deltas.where(deltas < 0, 0)

    # Calculate average gain and average loss
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def granger_causality_test(df, target_col, external_col, maxlag=12, verbose=True):
    """
    Performs Granger Causality tests for a target column against an external column in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        target_col (str): The name of the target column.
        external_col (str): The name of the external column.
        maxlag (int, optional): The maximum number of lags to include in the test. Default is 12.
        verbose (bool, optional): Whether to print the test results. Default is False.

    Returns:
        dict: A dictionary containing the test results for each lag, including F-statistic and p-value.
    """
    
    # Ensure the columns exist in the DataFrame
    if target_col not in df.columns or external_col not in df.columns:
        raise ValueError("Target or external column not found in DataFrame.")

    # Extract the data for the specified columns
    data = df[[target_col, external_col]].values

    # Perform Granger Causality tests
    results = grangercausalitytests(data, maxlag=maxlag, verbose=verbose)

    return results


def find_most_likely_distribution(target_df, target_column, bins=30):
    """
    Receives a Pandas DataFrame and a time series column
    and searches for the most similar statistical distribution.
    """

    sns.histplot(target_df[target_column], bins=30, kde=False, color='darkblue')
    plt.xlabel(target_column)
    plt.ylabel('Frequency')
    plt.title(f'{target_column} distribution')
    plt.show()

    # List of distributions to check
    distributions = [stats.norm, stats.lognorm, stats.expon, stats.gamma, stats.beta]

    # Fit distributions and calculate goodness-of-fit
    results = []
    for distribution in distributions:
        params = distribution.fit(target_df.dropna()[target_column])
        ks_stat, p_value = stats.kstest(target_df.dropna()[target_column], distribution.cdf, params)
        results.append((distribution.name, ks_stat, p_value))

    # Create a DataFrame to show the results
    df_results = pd.DataFrame(results, columns=['Distribution', 'KS Statistic', 'p-value'])
    print(df_results)

    # Determine the best fit based on the lowest KS statistic
    best_fit = df_results.loc[df_results['KS Statistic'].idxmin()]
    print(f"Best fitting distribution: {best_fit['Distribution']}")
    print(f"KS Statistic: {best_fit['KS Statistic']}")
    print(f"p-value: {best_fit['p-value']}")


def test_for_yearly_seasonality(df, value_column, date_column, alpha=0.05):

    # Group the data by month
    groups = [group[value_column].values for name, group in df.groupby(date_column)]

    # Perform the Kruskal-Wallis H test
    stat, p = stats.kruskal(*groups)

    print('\n', '-'*20, '[Kruskal-Wallis]', '-'*20)
    print(f'Statistic: {stat}')
    print(f'p-value: {p}')

    if p < alpha:
        print("There is a significant difference between the distributions of different months.")
    else:
        print("There is no significant difference between the distributions of different months.")

    # Perform the Bartlett test
    stat, p = stats.bartlett(*groups)

    print('\n', '-'*20, '[Barlett]', '-'*20)
    print(f'Statistic: {stat}')
    print(f'p-value: {p}')

    if p < alpha:
        print("There is a significant difference in variances between the distributions of different months.")
    else:
        print("There is no significant difference in variances between the distributions of different months.")




def calculate_bollinger_bands2(dataframe, target_column, window_size=20, num_std_dev=2):
    """
    Calculates Bollinger Bands for a target column in a Pandas DataFrame with a datetime index.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the target column and datetime index.
        target_column (str): The name of the column for which to calculate Bollinger Bands.
        window_size (int, optional): The rolling window size for calculating the moving average and standard deviation (default: 20).
        num_std_dev (int, optional): The number of standard deviations to use for the upper and lower bands (default: 2).

    Returns:
        pandas.DataFrame: The original DataFrame with three new columns:
            - 'middle_band': The moving average (middle line).
            - 'upper_band': The upper Bollinger Band.
            - 'lower_band': The lower Bollinger Band.
    """
    df = dataframe.copy()
    # Check if DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        #raise ValueError("DataFrame must have a datetime index.")
        df = df.set_index('date')

    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Calculate rolling mean and standard deviation
    rolling_mean = df[target_column].rolling(window=window_size, closed='left').mean()
    rolling_std = df[target_column].rolling(window=window_size, closed='left').std()

    # Calculate Bollinger Bands
    df[f'middle_band_{window_size}'] = rolling_mean
    df[f'upper_band_{window_size}'] = rolling_mean + (rolling_std * num_std_dev)
    df[f'lower_band_{window_size}'] = rolling_mean - (rolling_std * num_std_dev)
    df[f'bb_spread_{window_size}'] = df[f'upper_band_{window_size}'] - df[f'lower_band_{window_size}']
    df.reset_index(inplace=True)
    #df.drop(columns = [f'middle_band_{window_size}',f'upper_band_{window_size}', f'lower_band_{window_size}'], inplace=True)

    return df


def plot_bollinger_bands(df, target_column='close', window_size=20, num_std_dev=2):
    """
    Plots Bollinger Bands using Plotly.
    """

    df = calculate_bollinger_bands2(df, target_column, window_size, num_std_dev) # Assuming calculate_bollinger_bands is implemented

    fig = go.Figure()

    # Add trace for the price (close in this example)
    fig.add_trace(go.Scatter(x=df.index, y=df[target_column], name=target_column, line=dict(color=colors[0])))

    # Add traces for Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], name='Upper Band', line=dict(color=colors[3], dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['middle_band'], name='Middle Band', line=dict(color=colors[1])))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], name='Lower Band', line=dict(color=colors[2], dash='dash')))

    # Update layout
    fig.update_layout(
        title=f"Bollinger Bands ({target_column}, {window_size} weeks, {num_std_dev} std)",
        xaxis_title='Date',
        yaxis_title='Price',
        height=800
    )

    fig.show()


def infer_frequency(df):
    """
    Infer the frequency of the DataFrame's index.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with a datetime index.

    Returns:
    str: A string representation of the inferred frequency.
    """
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq is None:
        raise ValueError("Could not infer frequency from the DataFrame index.")
    return inferred_freq

def plot_fft(df, column, inferred_freq=None):
    """
    Compute and plot the Fast Fourier Transform (FFT) of the given column in the DataFrame.

    This function plots the amplitude of the frequency components and identifies significant
    cycles by plotting vertical reference lines for common cycles (monthly, quarterly, yearly,
    and 2-year cycles). It also identifies the strongest peak in the FFT and prints the
    corresponding cycle length in terms of the inferred frequency of the DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the time series data.
    column (str): The name of the column containing the target time series values.

    Returns:
    None
    """
    new_column = f'detrended_{column}'

    if inferred_freq is None:
        inferred_freq = infer_frequency(df)
    
    # Map the inferred frequency to a sample spacing in the same units as the index
    freq_map = {
        'H': 1,           # Hourly
        'D': 1,           # Daily
        'W': 1,           # Weekly
        'M': 1,           # Monthly
        'Q': 1,           # Quarterly
        'A': 1            # Yearly
    }
    
    if inferred_freq.startswith('W'):
        sample_spacing = 1.0
    elif inferred_freq.startswith('M'):
        sample_spacing = 1 / 4.33  # Average number of weeks in a month
    elif inferred_freq.startswith('Q'):
        sample_spacing = 1 / 13    # Average number of weeks in a quarter
    elif inferred_freq.startswith('A'):
        sample_spacing = 1 / 52    # Average number of weeks in a year
    elif inferred_freq.startswith('H'):
        sample_spacing = 1 / 24    # Number of hours in a day
    else:
        sample_spacing = freq_map.get(inferred_freq[0], 1)  # Default to 1 for unsupported frequencies

    df[new_column] = detrend(df[column])
    y = df[new_column].values
    y = df[new_column].values
    n = len(y)
    yf = fft(y)
    xf = fftfreq(n, sample_spacing)[:n//2]
    amplitude = 2.0/n * np.abs(yf[:n//2])
    
    plt.figure(figsize=(10, 6))
    plt.plot(xf, amplitude)
    plt.title(f'Fast Fourier Transform (FFT) for feature {column}')
    plt.xlabel('Frequency (1/{})'.format(inferred_freq))
    plt.ylabel('Amplitude')
    plt.grid()
    
    # Plot vertical lines for significant frequencies
    if inferred_freq.startswith('W'):
        plt.axvline(x=1/4.33, color='b', linestyle='--', label='Monthly cycle (1/4.33 weeks)')
        plt.axvline(x=1/13, color='m', linestyle='--', label='Quarterly cycle (1/13 weeks)')
        plt.axvline(x=1/52, color='r', linestyle='--', label='Yearly cycle (1/52 weeks)')
        plt.axvline(x=1/104, color='g', linestyle='--', label='2-year cycle (1/104 weeks)')
    elif inferred_freq.startswith('M'):
        plt.axvline(x=1/6, color='b', linestyle='--', label='Half Yearly cycle (1/6 months)')
        plt.axvline(x=1/12, color='r', linestyle='--', label='Yearly cycle (1/12 months)')
        plt.axvline(x=1/24, color='g', linestyle='--', label='2-year cycle (1/24 months)')
    elif inferred_freq.startswith('Q'):
        plt.axvline(x=1/2, color='b', linestyle='--', label='Half Yearly cycle (1/2 quarters)')
        plt.axvline(x=1/4, color='r', linestyle='--', label='Yearly cycle (1/4 quarters)')
        plt.axvline(x=1/8, color='g', linestyle='--', label='2-year cycle (1/8 quarters)')
    elif inferred_freq.startswith('H'):
        plt.axvline(x=1/(24), color='g', linestyle='--', label='Daily cycle (1/24 hours)')
        plt.axvline(x=1/(24*7), color='m', linestyle='--', label='Weekly cycle (1/168 hours)')
        plt.axvline(x=1/(24*30.44), color='b', linestyle='--', label='Monthly cycle (1/730.5 hours)')
        plt.axvline(x=1/(24*365.25), color='r', linestyle='--', label='Yearly cycle (1/8766 hours)')
        
    plt.legend()
    plt.show()
    
    # Find the strongest peak
    max_amplitude = np.max(amplitude[1:])  # Ignore the zero frequency component
    max_frequency = xf[np.argmax(amplitude[1:]) + 1]  # +1 to account for ignoring zero frequency
    
    # Calculate the cycle length
    cycle_length = 1 / max_frequency
    
    print(f"The strongest cycle has a frequency of {max_frequency:.5f} (1/{inferred_freq}), "
          f"corresponding to a cycle length of approximately {cycle_length:.2f} {inferred_freq}.")


def align_to_weekday_resample(df, date_col='date', target_col='value', desired_weekday='MON'):
    """
    Aligns a time series with irregular weekdays to the desired weekday using resampling.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame with the time series data.
        date_col (str): The name of the column containing datetime values.
        target_col (str): The name of the column containing the time series values.
        desired_weekday (str or int): The desired weekday to align to (e.g., 'MON' for Monday, 0 for Monday, etc.).
    
    Returns:
        pandas.DataFrame: A new DataFrame with the time series aligned to the desired weekday.
    """

    # Ensure correct data types
    df[date_col] = pd.to_datetime(df[date_col])

    # Create a copy to avoid modifying original data
    df_aligned = df.copy()
    
    # Set the date column as the index
    df_aligned.set_index(date_col, inplace=True)

    # Resample to weekly frequency starting on the desired weekday
    df_aligned = df_aligned.resample(f'W-{desired_weekday}').first()  

    # Reset index
    df_aligned.reset_index(inplace=True)
    df_aligned.date = pd.to_datetime(df_aligned['date'])

    return df_aligned


def plot_series_2yaxis(df, target, x_label,date_col):
    fig = go.Figure()

    # Plotting Suzano's average price

    for i,j in enumerate(x_label):
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[j],
            name=f'{j}',
            mode='lines+markers',
            connectgaps=True,
            yaxis='y2',
            line=dict(color=colors[i+2])
        ))

    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[target],
        name=f'{target}',
        mode='lines+markers',
        line=dict(color=colors[0])
    ))

    fig.update_layout(
            title={
            'text': f'Comparing the {target} with the {x_label[0]}',
            'font': {'size': 24}
        },
        xaxis_title={
            'text': 'Date',
            'font': {'size': 20}
        },
        yaxis_title={
            'text': f'{target}',
            'font': {'size': 20}
        },
        yaxis2=dict(
                    title={
                'text': f'{x_label[0]}',
                'font': {'size': 20}
            },
            overlaying='y',
            side='right'
        ),
        legend=dict(
            x=0.1,
            y=1.1,
            traceorder='normal',
            orientation='h',
            font=dict(size=16)
        ),
        height=600,
        width=1200
    )

    # Show plot
    fig.show()


def spearman_correlation_test(df, covariable, target):

    df_filtered = df[[covariable, target]].dropna().copy()

    spearman_corr, p_value = stats.spearmanr(df_filtered[covariable], df_filtered[target])
    if p_value < 0.05:
        res = 'Significativo!'
    else:
        res = 'Não Significativo!'
    print(f"{covariable}:\n")
    print(f"Spearman correlation coefficient: {spearman_corr}")
    print(f"P-value: {round(p_value,2)} | {res}")
    print('-----------------------------------\n')


def pearson_correlation_test(df, covariable, target):

    df_filtered = df[[covariable, target]].dropna().copy()

    spearman_corr, p_value = stats.pearsonr(df_filtered[covariable], df_filtered[target])
    if p_value < 0.05:
        res = 'Significativo!'
    else:
        res = 'Não Significativo!'
    print(f"{covariable}:\n")
    print(f"Pearson correlation coefficient: {spearman_corr}")
    print(f"P-value: {round(p_value,2)} | {res}")
    print('-----------------------------------\n')


def create_subplots(df):

    # Create subplots
    num_columns = len(df.columns)
    num_rows = (num_columns + 1) // 2  # Calculate rows for 2 columns layout

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    # Loop through each column and create a subplot
    for i, column in enumerate(df.columns):
        df.plot.scatter(y='order_intake_enter', x=column, ax=axs[i], title=f'{column} vs order_intake_enter')

    # Turn off unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def evaluate_linear_model_split(df, target_column, test_size=0.2):
    # Check if target_column exists in DataFrame
    if target_column not in df.columns:
        raise ValueError(f"{target_column} is not a column in the DataFrame.")
    
    # Prepare data
    X = df.drop(columns=[target_column])  # Independent variables
    y = df[target_column]  # Dependent variable
    X = sm.add_constant(X)  # Add constant for intercept
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f'train set dimension: {X_train.shape}')
    print(f'test set dimension: {X_test.shape}')
    
    # Fit a linear model on the training set
    model = sm.OLS(y_train, X_train).fit()
    
    # Print model summary
    print(model.summary())
    
    # Print coefficients
    print("\nCoefficients:")
    for variable, coef in zip(X.columns, model.params):
        print(f"{variable}: {coef:.4f}")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate RMSE
    rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    rmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
    
    print(f'\nRoot Mean Square Error (RMSE) - Train: {rmse_train:.4f}')
    print(f'Root Mean Square Error (RMSE) - Test: {rmse_test:.4f}')
    
    # Calculate MAPE
    mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    print(f'Mean Absolute Percentage Error (MAPE) - Train: {mape_train:.2f}%')
    print(f'Mean Absolute Percentage Error (MAPE) - Test: {mape_test:.2f}%')
    
    # Residuals
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred
    
    # Normality test for residuals
    shapiro_train = stats.shapiro(residuals_train)
    shapiro_test = stats.shapiro(residuals_test)
    
    print(f'\nShapiro-Wilk Test - Train: Statistic={shapiro_train.statistic:.4f}, p-value={shapiro_train.pvalue:.4f}')
    print(f'Shapiro-Wilk Test - Test: Statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}')
    
    # Plotting residuals
    plt.figure(figsize=(16, 12))

    # Train residuals vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_train_pred, residuals_train)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Train Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    # Histogram of Train residuals
    plt.subplot(2, 2, 2)
    plt.hist(residuals_train, bins=20, edgecolor='black')
    plt.title('Histogram of Train Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    
    # Test residuals vs Predicted
    plt.subplot(2, 2, 3)
    plt.scatter(y_test_pred, residuals_test)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Test Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    # Histogram of Test residuals
    plt.subplot(2, 2, 4)
    plt.hist(residuals_test, bins=20, edgecolor='black')
    plt.title('Histogram of Test Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    
    # Show plots
    plt.tight_layout()
    plt.show()



def evaluate_linear_model(df, target_column, test_size=0.2):
    # Check if target_column exists in DataFrame
    if target_column not in df.columns:
        raise ValueError(f"{target_column} is not a column in the DataFrame.")
    
    # Prepare data
    X = df.drop(columns=[target_column])  # Independent variables
    y = df[target_column]  # Dependent variable
    X = sm.add_constant(X)  # Add constant for intercept
    
    
    # Fit a linear model on the training set
    model = sm.OLS(y, X).fit()
    
    # Print model summary
    print(model.summary())
    
    # Print coefficients
    print("\nCoefficients:")
    for variable, coef in zip(X.columns, model.params):
        print(f"{variable}: {coef:.4f}")
    
    # Predictions
    y_pred = model.predict(X)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    print(f'\nRoot Mean Square Error (RMSE) : {rmse:.4f}')

    
    # Calculate MAPE
    mape = np.mean(np.abs((y - y_pred) / y)) * 100

    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    
    # Residuals
    residuals = y- y_pred
    
    # Normality test for residuals
    shapiro = stats.shapiro(residuals)

    print(f'\nShapiro-Wilk Test: Statistic={shapiro.statistic:.4f}, p-value={shapiro.pvalue:.4f}')
    
    
    # Plotting residuals
    plt.figure(figsize=(16, 12))

    # Train residuals vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    # Histogram of Train residuals
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    
    
    # Show plots
    plt.tight_layout()
    plt.show()


# Definindo a função customizada para o CatBoost
def catboost_asymmetric_loss(y_true, y_pred):
    residual = y_true - y_pred
    loss = np.where(residual > 0, 2.0 * residual**2, residual**2)
    grad = np.where(residual > 0, -4.0 * residual, -2.0 * residual)
    hess = np.full_like(residual, 2.0)
    return loss, grad, hess

# Função de perda customizada para o LightGBM
def lightgbm_asymmetric_loss(y_true, y_pred):
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2.0 * 2.0 * residual, -2.0 * residual)
    hess = np.where(residual > 0, 2.0 * 2.0, 2.0)
    return grad, hess

# Função de perda customizada para o XGBoost
def xgboost_asymmetric_loss(y_true, y_pred):
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2.0 * 2.0 * residual, -2.0 * residual)
    hess = np.where(residual > 0, 2.0 * 2.0, 2.0)
    return grad, hess
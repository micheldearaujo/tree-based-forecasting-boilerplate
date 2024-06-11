import sys
sys.path.insert(0,'.')

import re
import warnings
from src.utils import *
from src.features.feat_eng import build_features
from src.config import features_list

warnings.filterwarnings("ignore")
logger = logging.getLogger("inference")
logger.setLevel(logging.DEBUG)

def make_future_df(forecast_horzion: int, model_df: pd.DataFrame, features_list: list) -> pd.DataFrame:
    """
    Create a future dataframe for forecasting.

    Parameters:
        forecast_horizon (int): The number of days to forecast into the future.
        model_df (pandas dataframe): The dataframe containing the training data.

    Returns:
        future_df (pandas dataframe): The future dataframe used for forecasting.
    """

    # create the future dataframe with the specified number of days
    last_training_day = model_df.Date.max()
    date_list = [last_training_day + dt.timedelta(days=x+1) for x in range(forecast_horzion+1)]
    future_df = pd.DataFrame({"Date": date_list})
    
    # add stock column to iterate
    future_df["Stock"] = model_df.Stock.unique()[0]

    # build the features for the future dataframe using the specified features
    inference_features_list = [feature for feature in features_list if "MA" not in feature and "lag" not in feature]
    future_df = build_features(future_df, inference_features_list, save=False)

    # filter out weekends from the future dataframe
    future_df["day_of_week_name"] = future_df.Date.apply(lambda x: x.day_name())
    future_df = future_df[future_df["day_of_week_name"].isin(["Sunday", "Saturday"]) == False]
    future_df = future_df.drop("day_of_week_name", axis=1)
    future_df = future_df.reset_index(drop=True)
    
    # set the first lagged price value to the last price from the training data
    ma_and_lag_features = [feature for feature in features_list if feature not in inference_features_list]
    for feature in ma_and_lag_features:
        
        future_df[feature] = 0
        if "lag" in feature:
            lag_value = int(feature.split("_")[-1])
            future_df.loc[future_df.index.min(), feature] = model_df['Close'].values[-lag_value]
        else:
            ma_value = int(feature.split("_")[-1])
            future_df.loc[future_df.index.min(), feature] = model_df['Close'].rolling(ma_value).mean().values[-1]

    return future_df


def make_predict(model, forecast_horizon: int, future_df: pd.DataFrame, past_target_values: list) -> pd.DataFrame:

    """
    Make predictions for the next `forecast_horizon` days using a XGBoost model
    
    Parameters:
        model (sklearn model): Scikit-learn best model to use to perform inferece.
        forecast_horizon (pandas dataframe): The amount of days to predict into the future.
        future_df (pd.DataFrame): The "Feature" DataFrame (X) with future index.
        past_df (pd.DataFrame): The past values DataFrame, to calculate the moving averages on.
        
    Returns:
        pd.DataFrame: The future DataFrame with forecasts.
    """

    future_df_feat = future_df.copy()

    # Create empty list for storing each prediction
    predictions = []

    updated_fh = len(future_df_feat) # after removing weekends

    for day in range(0, updated_fh):

        # extract the next day to predict
        x_inference = pd.DataFrame(future_df_feat.drop(columns=["Date", "Stock"]).loc[day, :]).transpose()
        prediction = model.predict(x_inference)[0]
        predictions.append(prediction)
        all_features = future_df_feat.columns

        # Append the prediction to the last_closing_prices
        past_target_values.append(prediction)

        # get the prediction and input as the lag 1
        if day != updated_fh-1:
            
            lag_features = [feature for feature in all_features if "lag" in feature]
            for feature in lag_features:
                lag_value = int(feature.split("_")[-1])
                future_df_feat.loc[day+1, feature] = past_target_values[-lag_value]

            
            moving_averages_features = [feature for feature in all_features if "MA" in feature]
            for feature in moving_averages_features:
                ma_value = int(feature.split("_")[-1])
                last_n_closing_prices = [*past_target_values[-ma_value+1:], prediction]
                next_ma_value = np.mean(last_n_closing_prices)
                future_df_feat.loc[day+1, feature] = next_ma_value

        else:
            # check if it is the last day, so we stop
            break
    
    future_df_feat["Forecast"] = predictions
    future_df_feat["Forecast"] = future_df_feat["Forecast"].astype('float64')
    future_df_feat = future_df_feat[["Date", "Forecast"]].copy()
    
    return future_df_feat


def load_production_model(logger, model_config, stock_name):
    """
    Search for models in "Production" state using Mlflow's API, given a model name and a stock name.
    """

    client = MlflowClient()

    logger.info(f"Searching for production models for stock {stock_name}...")
    models_versions = []

    for mv in client.search_model_versions("name='{}_{}'".format(model_config[f'REGISTER_MODEL_NAME_INF'], stock_name)):
        models_versions.append(dict(mv))

    try:
        logger.debug("Previous model version found. Loading it...") 
        current_prod_inf_model = [x for x in models_versions if x['current_stage'] == 'Production'][0]
        current_model_path = current_prod_inf_model['source']
        current_model_path = re.search(r'mlruns.*/model', current_model_path)
        current_model_path = current_model_path.group()

        current_model_path = current_model_path[:-5] + 'xgboost_model_' + stock_name
        logger.debug(f"Loading model from path: \n{current_model_path}")
        current_prod_model = mlflow.xgboost.load_model(model_uri='./'+current_model_path)

        return current_prod_model

    except IndexError:
        logger.warning("NO PRODUCTION MODEL FOUND. STOPPING THE PIPELINE!")
        return None


def inference_pipeline():
    """
    Run the inference pipeline for predicting stock prices using the production model.

    This function loads the featurized dataset, searches for the production model for a specific stock,
    and makes predictions for the future timeframe using the loaded model. The predictions are then
    saved to a CSV file.

    Parameters:
        None

    Returns:
        None
    """


    logger.debug("Loading the featurized dataset...")
    stock_df_feat_all = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])
    
    final_predictions_df = pd.DataFrame()

    for stock_name in stock_df_feat_all["Stock"].unique():
        logger.info(f"Performing inferece for ticker symbol {stock_name}...")

        stock_df_feat = stock_df_feat_all[stock_df_feat_all["Stock"] == stock_name].copy()
        current_prod_model = load_production_model(logger, model_config, stock_name)

        logger.debug("Creating the future dataframe...")
        future_df = make_future_df(model_config["FORECAST_HORIZON"], stock_df_feat, features_list)
        
        logger.debug("Predicting...")
        predictions_df = make_predict(
            model=current_prod_model,
            forecast_horizon=model_config["FORECAST_HORIZON"]-4,
            future_df=future_df,
            past_target_values=list(stock_df_feat['Close'].values)
        )

        predictions_df["Stock"] = stock_name
        
        final_predictions_df = pd.concat([final_predictions_df, predictions_df], axis=0)


    logger.debug("Writing the predictions to database...")
    final_predictions_df.to_csv(os.path.join(OUTPUT_DATA_PATH, 'output_stock_prices.csv'), index=False)

    logger.debug("Predictions written successfully!")


# Execute the whole pipeline
if __name__ == "__main__":

    logger.info("Starting the Inference pipeline...")

    inference_pipeline()

    logger.info("Inference Pipeline was successful!\n")


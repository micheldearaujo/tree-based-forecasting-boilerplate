# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

from src.utils import *
from src.config import param_distributions_dict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger("model-testing")
logger.setLevel(logging.INFO)


def tune_model_hyperparameters(model, X_train, y_train, param_distributions, n_iter=500, cv=5, scoring=None):
    """
    Tune model hyperparameters using RandomizedSearchCV.
    
    Parameters:
    - model: The machine learning model to tune.
    - X_train: Training features.
    - y_train: Training target variable.
    - param_distributions: Dictionary with parameters names (`str`) as keys and distributions
                           or lists of parameters to try.
    - n_iter: Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    - cv: Cross-validation splitting strategy.
    - scoring: A single string or callable to evaluate the predictions on the test set. If None, the model's default scorer is used.
    
    Returns:
    - best_model: The tuned model with the best parameters.
    - results: The results of the RandomizedSearchCV.
    """
    
    # If no custom scoring function is provided, use mean squared error
    if scoring is None:
        scoring = make_scorer(mean_squared_error, greater_is_better=False)

    model.set_params(**{'n_jobs': 1})
    random_search = RandomizedSearchCV(
        model, param_distributions=param_distributions,
        n_iter=n_iter, cv=cv, scoring=scoring, verbose=1, n_jobs=-1, random_state=42
    )
    
    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)
    
    # Best model
    best_model = random_search.best_estimator_
    
    # Results
    results = {
        'Best Parameters': random_search.best_params_,
        'Best Score': random_search.best_score_
    }
    
    return best_model, results


def calculate_adjusted_forecast_horizon(original_forecast_horizon, weekend_days_per_week):

    adjusted_forecast_horizon = original_forecast_horizon - int(original_forecast_horizon * weekend_days_per_week / 7)

    return adjusted_forecast_horizon


def time_series_grid_search_xgb(X, y, param_grid: dict, stock_name, n_splits=5, random_state=0):
    """
    Performs time series hyperparameter tuning on an XGBoost model using grid search.
    
    Args:
        X (pd.DataFrame): The input feature data
        y (pd.Series): The target values
        param_grid (dict): Dictionary of hyperparameters to search over
        n_splits (int): Number of folds for cross-validation (default: 5)
        random_state (int): Seed for the random number generator (default: 0)
    
    Returns:
        tuple: A tuple containing the following elements:
            best_model (xgb.XGBRegressor): The best XGBoost model found by the grid search
            best_params (dict): The best hyperparameters found by the grid search
    """

    # perform time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = xgb.XGBRegressor(random_state=random_state)
    grid_search = GridSearchCV(model, param_grid, cv=tscv, n_jobs=-1, scoring='neg_mean_absolute_error', verbose=1)
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # save the best model
    dump(best_model, f"./models/{stock_name}_xgb.joblib")
    return best_model, best_params


def test_model_one_shot(X: pd.DataFrame, y: pd.Series, forecast_horizon: int, stock_name: str, tunning: bool = False, refit: bool = False) -> pd.DataFrame:
    """
    Make predictions for the past `forecast_horizon` days using a XGBoost model.
    This model is validated using One Shot Training, it means that we train the model
    once, and them perform the `forecast_horizon` predictions only loading the mdoel.
    
    Parameters:
        X (pandas dataframe): The input data
        y (pandas Series): The target data
        forecast_horizon (int): Number of days to forecast

    Returns:
        pred_df: Pandas DataFrame with the forecasted values
    """

    # Create empty list for storing each prediction
    predictions = []
    actuals = []
    dates = []
    X_testing_df = pd.DataFrame()
    
    forecast_horizon_adj = calculate_adjusted_forecast_horizon(forecast_horizon, 2)
    # forecast_horizon_adj = 5

    # get the one-shot training set
    X_train = X.iloc[:-forecast_horizon_adj, :]
    y_train = y.iloc[:-forecast_horizon_adj]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=0)

    final_y = y_train.copy()
    final_X = X_train.copy()

    # print("Initial Training Shape: ", X_train.shape)
    # print("Initial Training Target Shape: ", y_train.shape)
    
    # print("Last Training Date: ", X_train['Date'].max())
    # print("Last Training Target: ", y_train.values[-1])
    params = {'alpha': 0.15, 'gamma': 0.15, 'lambda': 0.15, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}  # - Better
    # params = {'alpha': 0.05, 'gamma': 0.05, 'lambda': 0.05, 'learning_rate': 0.2, 'max_depth': 2, 'n_estimators': 200}

    best_model = xgb.XGBRegressor(
        eval_metric=["rmse"],
        tree_method="hist",
        **params,
    )

    if tunning:
        logger.debug("HyperTunning the model...")

        best_model, results = tune_model_hyperparameters(best_model, X_train.drop(columns=["Date"]), y_train, param_distributions_dict[str(type(best_model)).split('.')[-1][:-2]], cv=5)

    else:
        
        logger.debug("Fitting the model without HyperTunning...")
        best_model.fit(
            X_train.drop("Date", axis=1),
            y_train,
            eval_set=[(X_train.drop("Date", axis=1), y_train), (X_val.drop("Date", axis=1), y_val)],
            verbose=50
        )

    # Iterate over the dataset to perform predictions over the forecast horizon, one by one.
    # After forecasting the next step, we need to update the "lag" features with the last forecasted
    # value
    for day in range(forecast_horizon_adj, 0, -1):

        if refit and day != forecast_horizon_adj:
                
            print("Refitting the model...")
            print(f"Dataset Atualizado data Máxima: {final_X['Date'].max()}")
            X_train_new = final_X.iloc[:-day, :]
            y_train_new = final_y.iloc[:-day]

            print("Refitting Training Shape: ", X_train_new.shape)
            print("Refitting Last Training Date: ", X_train_new['Date'].max())
            print("Refitting Training Target Shape: ", y_train_new.shape)
            
            X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_train_new, y_train_new, test_size=0.05, random_state=0)

            # Train model again
            best_model = xgb.XGBRegressor(
                eval_metric=["rmse"],
                **params,
                tree_method="hist",
                verbose=50
            )

            best_model.fit(
                X_train_new.drop("Date", axis=1),
                y_train_new,
                eval_set=[(X_train_new.drop("Date", axis=1), y_train_new), (X_val_new.drop("Date", axis=1), y_val_new)],
                verbose=0
            )

        if day != 1:
            # the testing set will be the next day after the training and we use the complete dataset
            X_test = X.iloc[-day:-day+1,:]
            y_test = y.iloc[-day:-day+1]

        else:
            # need to change the syntax for the last day (for -1:-2 will not work)
            X_test = X.iloc[-day:,:]
            y_test = y.iloc[-day:]
            
        # print(f"Day: {day} | Testing date: {X_test['Date'].values[0]} | Dia da Semana: {X_test.Date.apply(lambda x: x.day_name())}")


        # only the first iteration will use the true value of Close_lag_1
        # because the following ones will use the last predicted value
        # so we simulate the process of predicting out-of-sample
        if len(predictions) != 0:
            
            lag_features = [feature for feature in X_test.columns if "lag" in feature]
            for feature in lag_features:
                lag_value = int(feature.split("_")[-1])
                index_to_replace = list(X_test.columns).index(feature)
                X_test.iat[0, index_to_replace] = final_y.iloc[-lag_value]


            moving_averages_features = [feature for feature in X_test.columns if "MA" in feature]
            for feature in moving_averages_features:
                ma_value = int(feature.split("_")[-1])
                last_closing_princes_ma = final_y.rolling(ma_value).mean()
                last_ma = last_closing_princes_ma.values[-1]
                index_to_replace = list(X_test.columns).index(feature)
                X_test.iat[0, index_to_replace] = last_ma

            X_testing_df = pd.concat([X_testing_df, X_test], axis=0)

        else:
            # we jump the first iteration because we do not need to update anything.
            X_testing_df = pd.concat([X_testing_df, X_test], axis=0)
            pass

        # make prediction
        prediction = best_model.predict(X_test.drop("Date", axis=1))

        # Append the both Prediction and updated X_test to the final lists
        final_y = pd.concat([final_y, pd.Series(prediction[0])], axis=0)
        final_y = final_y.reset_index(drop=True)

        final_X = pd.concat([final_X, X_test], axis=0)
        final_X = final_X.reset_index(drop=True)

        # store the results
        predictions.append(prediction[0])
        actuals.append(y_test.values[0])
        dates.append(X_test["Date"].max())

    pred_df = pd.DataFrame(list(zip(dates, actuals, predictions)), columns=["Date", 'Actual', 'Forecast'])
    pred_df["Forecast"] = pred_df["Forecast"].astype("float64")

    logger.debug("Calculating the evaluation metrics...")
    model_mape = round(mean_absolute_percentage_error(actuals, predictions), 4)
    model_rmse = round(np.sqrt(mean_squared_error(actuals, predictions)), 2)
    model_mae = round(mean_absolute_error(actuals, predictions), 2)
    model_wape = round((pred_df.Actual - pred_df.Forecast).abs().sum() / pred_df.Actual.sum(), 2)

    pred_df["MAPE"] = model_mape
    pred_df["MAE"] = model_mae
    pred_df["WAPE"] = model_wape
    pred_df["RMSE"] = model_rmse
    pred_df["Model"] = str(type(best_model)).split('.')[-1][:-2]

    return pred_df, X_testing_df


def model_testing_pipeline(tunning=False):

    logger.debug("Loading the featurized dataset..")
    stock_df_feat_all = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])

    # iterate over the stocks
    validation_report_df = pd.DataFrame()

    for stock_name in stock_df_feat_all["Stock"].unique():

        logger.info("Testing the model for the stock: %s..."%stock_name)
        stock_df_feat = stock_df_feat_all[stock_df_feat_all["Stock"] == stock_name].copy().drop("Stock", axis=1)
        
        predictions_df, X_testing_df = test_model_one_shot(
            X=stock_df_feat.drop([model_config["TARGET_NAME"]], axis=1),
            y=stock_df_feat[model_config["TARGET_NAME"]],
            forecast_horizon=model_config['FORECAST_HORIZON'],
            stock_name=stock_name,
            tunning=tunning
        )

        predictions_df["Stock"] = stock_name
        predictions_df["Training_Date"] = dt.datetime.today().date()

        validation_report_df = pd.concat([validation_report_df, predictions_df], axis=0)
    
    logger.debug("Writing the testing results dataframe...")
    validation_report_df = validation_report_df.rename(columns={"Forecast": "Price"})
    validation_report_df["Class"] = "Testing"

    logger.info(f"Median MAPE: {validation_report_df['MAPE'].median()}")

    validation_report_df.to_csv(os.path.join(OUTPUT_DATA_PATH, 'validation_stock_prices.csv'), index=False)


# Execute the whole pipeline
if __name__ == "__main__":
    logger.info("Starting the Model Testing pipeline...")

    model_testing_pipeline(tunning=False)
    
    logger.info("Model Testing Pipeline was sucessful!")
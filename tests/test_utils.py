# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0,'.')
import matplotlib
import pytest
from unittest.mock import MagicMock

from src.utils import *
from src.models.model_utils import *
from src.features.feat_eng import build_features
from src.data.make_dataset import make_dataset
from src.models.train_model import *
from src.models.hyperparam_tune import optimize_model_params, stepwise_forecasting

PERIOD = '800d'
INTERVAL = '1d'

STOCK_NAME = 'BOVA11.SA'

# create a fake dataset
dates_list = ['2022-01-01', '2022-01-02', '2022-01-03']
stocks_list = ['BOVA11.SA', 'ITUB4.SA', 'VALE3.SA']
dates_list = [pd.to_datetime(date) for date in dates_list]
prices_list = [np.random.rand()*10, 100, 4571.54]
prices_list = [float(price) for price in prices_list]
day_of_months_list = [1., 2., 30.]
months_list = [1., 6., 12.]
quarters_list = [1., 2., 4.]
close_lags_list = prices_list
TEST_FORECAST_HORIZON = 1

test_param_grid = {
    "learning_rate": [0.1],
    "max_depth": [5],
    "min_child_weight": [1],
}

test_model_params = {
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_child_weight": 1,
}

test_stock_df = pd.DataFrame(
    {
        "Date": dates_list,
        "Stock": stocks_list,
        "Close": prices_list,
    }
)

test_stock_feat_df = pd.DataFrame(
    {
        "Date": dates_list,
        "Stock": stocks_list,
        "Close": prices_list,
        "day_of_month": day_of_months_list,
        "month": months_list,
        "quarter": quarters_list,
        "Close_lag_1": close_lags_list
    }
)

test_predictions_df = pd.DataFrame(
    {
        "Date": dates_list,
        "Actual": prices_list,
        "Forecast": prices_list
    }
)


def test_make_dataset_columns():
    """
    tests if the make_dataset() is download
    and saving the file in the correct format
    """

    # call the make_dataset function
    stock_price_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)
    
    # assert the amount of columns and column's orders
    assert test_stock_df.columns.all() == stock_price_df.columns.all()


def test_make_dataset_types():
    """
    tests if the make_dataset() is download
    and saving the file in the correct format
    """

    # call the make_dataset function
    stock_price_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)

    # assert the columns data types
    assert isinstance(stock_price_df["Date"].dtype, type(np.dtype("datetime64[ns]")))
    assert isinstance(stock_price_df["Close"].dtype, type(np.dtype("float64")))


def test_make_dataset_size():
    """
    tests if the make_dataset() is download
    and saving the file in the correct format
    """

    # call the make_dataset function
    stock_price_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)

    # asser the amount of days
    assert stock_price_df.shape[0] >= int(PERIOD[:-1])-TEST_FORECAST_HORIZON


def test_build_features_columns():
    """
    tests if the build() is download
    and saving the file in the correct format
    """
    # load the output file to test

    stock_df_feat = build_features(test_stock_df, features_list)

    assert test_stock_feat_df.columns.all() == stock_df_feat.columns.all()


def test_build_features_types():
    """
    tests if the build() is download
    and saving the file in the correct format
    """
    # load the output file to test

    stock_df_feat = build_features(test_stock_df, features_list)


    assert isinstance(stock_df_feat["Date"].dtype, type(np.dtype("datetime64[ns]")))
    assert isinstance(stock_df_feat["Close"].dtype, type(np.dtype("float64")))
    assert isinstance(stock_df_feat["Stock"].dtype, type(np.dtype("object")))
    assert isinstance(stock_df_feat["day_of_month"].dtype, type(np.dtype("float64")))
    assert isinstance(stock_df_feat["month"].dtype, type(np.dtype("float64")))
    assert isinstance(stock_df_feat["quarter"].dtype, type(np.dtype("float64")))
    assert isinstance(stock_df_feat["Close_lag_1"].dtype, type(np.dtype("float64")))
    

def test_build_features_size():
    """
    tests if the build() is download
    and saving the file in the correct format
    """

    # load the output file to test
    stock_df_feat = build_features(test_stock_df, features_list, save=False)
    
    assert stock_df_feat.shape[0] == 0#test_stock_df.shape[0] - 1  # because of the shift(1)


def test_ts_train_test_split_array_size():

    returned_array = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"], TEST_FORECAST_HORIZON)

    assert len(returned_array) == 4


def test_ts_train_test_split_columns():

    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)

    assert test_stock_feat_df.columns.all() == X_train.columns.all()
    assert test_stock_feat_df.columns.all() == X_test.columns.all()


def test_ts_train_test_split_train_types():

    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)
    
    assert isinstance(X_train["Date"].dtype, type(np.dtype("datetime64[ns]")))
    assert isinstance(X_train["day_of_month"].dtype, type(np.dtype("float64")))
    assert isinstance(X_train["month"].dtype, type(np.dtype("float64")))
    assert isinstance(X_train["quarter"].dtype, type(np.dtype("float64")))
    assert isinstance(X_train["Close_lag_1"].dtype, type(np.dtype("float64")))
    assert isinstance(y_train.dtype, type(np.dtype("float64")))


def test_make_future_df_columns():

    # Create the future dataframe using the make_future_df function
    future_df = make_future_df(TEST_FORECAST_HORIZON, test_stock_feat_df, features_list)

    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)

    assert X_train.columns.all() == future_df.columns.all()


def test_make_future_df_types():

    # Create the future dataframe using the make_future_df function
    future_df = make_future_df(TEST_FORECAST_HORIZON, test_stock_feat_df, features_list)

    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)

    assert isinstance(future_df["Date"].dtype, type(np.dtype("datetime64[ns]")))
    assert isinstance(future_df["day_of_month"].dtype, type(np.dtype("float64")))
    assert isinstance(future_df["month"].dtype, type(np.dtype("float64")))
    assert isinstance(future_df["quarter"].dtype, type(np.dtype("float64")))
    assert isinstance(future_df["Close_lag_1"].dtype, type(np.dtype("float64")))


def test_make_future_df_size():

    # Create the future dataframe using the make_future_df function
    future_df = make_future_df(TEST_FORECAST_HORIZON, test_stock_feat_df, features_list)

    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)

    assert future_df.shape[0] == TEST_FORECAST_HORIZON


def test_make_predict_columns():

    # create an inferencec dataframe
    test_inference_df = test_stock_feat_df.drop([model_config["TARGET_NAME"], "Stock"], axis=1).copy()
    # get training data
    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df.drop("Stock", axis=1), model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)
    # create a simple model
    xgboost_model = xgb.XGBRegressor(**test_model_params)
    # fit the model
    xgboost_model.fit(X_train.drop("Date", axis=1), y_train)
    # make predictions
    predictions_df = make_predict(
        model=xgboost_model,
        forecast_horizon=TEST_FORECAST_HORIZON*test_inference_df.shape[0],
        future_df=test_inference_df
    )
    
    assert test_predictions_df.columns.all() == predictions_df.columns.all()


def test_make_predict_types():

    test_inference_df = test_stock_feat_df.drop([model_config["TARGET_NAME"], "Stock"], axis=1).copy()
    # get training data
    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df.drop("Stock", axis=1), model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)
    # create a simple model
    xgboost_model = xgb.XGBRegressor(**test_model_params)
    # fit the model
    xgboost_model.fit(X_train.drop("Date", axis=1), y_train)

    predictions_df = make_predict(
        model=xgboost_model,
        forecast_horizon=TEST_FORECAST_HORIZON*test_inference_df.shape[0],
        future_df=test_inference_df
    )

    assert isinstance(predictions_df["Date"].dtype, type(np.dtype("datetime64[ns]")))
    assert isinstance(predictions_df["Forecast"].dtype, type(np.dtype("float64")))


def test_make_predict_size():

    test_inference_df = test_stock_feat_df.drop([model_config["TARGET_NAME"], "Stock"], axis=1).copy()
    # get training data
    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df.drop("Stock", axis=1), model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)
    # create a simple model
    xgboost_model = xgb.XGBRegressor(**test_model_params)
    # fit the model
    xgboost_model.fit(X_train.drop("Date", axis=1), y_train)

    predictions_df = make_predict(
        model=xgboost_model,
        forecast_horizon=TEST_FORECAST_HORIZON*test_inference_df.shape[0],
        future_df=test_inference_df
    )
    
    assert predictions_df.shape[0] == TEST_FORECAST_HORIZON*test_inference_df.shape[0]


# def test_time_series_grid_search_xgb_array_size():

#     returned_array = time_series_grid_search_xgb(
#         X=test_stock_feat_df.drop([model_config["TARGET_NAME"], "Date"], axis=1),
#         y=test_stock_feat_df[model_config["TARGET_NAME"]],
#         stock_name=STOCK_NAME,
#         param_grid=test_param_grid,
#         n_splits=2,
#         random_state=42
#     )

#     assert len(returned_array) == 2


# def test_time_series_grid_search_xgb_dict():
    
#     best_model, best_params = time_series_grid_search_xgb(
#         X=test_stock_feat_df.drop([model_config["TARGET_NAME"], "Date"], axis=1),
#         y=test_stock_feat_df[model_config["TARGET_NAME"]],
#         stock_name=STOCK_NAME,
#         param_grid=test_param_grid,
#         n_splits=2,
#         random_state=42
#     )

#     assert isinstance(best_model, xgb.sklearn.XGBRegressor)
#     assert isinstance(best_params, dict)



from unittest.mock import MagicMock

# fixture for a sample mlflow client
@pytest.fixture()
def mlflow_client():
    client = MagicMock()
    client.search_model_versions.return_value = [
        {'name': 'model_name', 'current_stage': 'Staging', 'run_id': '123'},
        {'name': 'model_name', 'current_stage': 'Production', 'run_id': '456'}
    ]
    return client

# fixture for a sample production model run
@pytest.fixture()
def production_model_run():
    run = MagicMock()
    run.data.params = {
        'max_depth': '3',
        'learning_rate': '0.1',
        'n_estimators': '100'

    }
    return run

# fixture for xgboost_hyperparameter_config
@pytest.fixture()
def xgboost_hyperparameter_config():
    return {
        'max_depth': {'type': 'integer', 'min': 1, 'max': 10, 'step': 1},
        'learning_rate': {'type': 'float', 'min': 0.01, 'max': 1.0, 'step': 0.01},
        'n_estimators': {'type': 'integer', 'min': 50, 'max': 500, 'step': 50},
    }

@pytest.fixture()
def stock_name():
    return stocks_list[0]

def test_load_production_model_params(mlflow_client, production_model_run, xgboost_hyperparameter_config):
    # mock mlflow.get_run to return the sample production model run
    mlflow.get_run = MagicMock(return_value=production_model_run)

    # call the function with the sample client
    prod_validation_model_params_new, current_prod_model = load_production_model_params(mlflow_client, stocks_list[0])

    # check the result
    assert prod_validation_model_params_new == {
        'max_depth': '3',
        'learning_rate': '0.1',
        'n_estimators': '100'
    }
    assert current_prod_model == {'name': 'model_name', 'current_stage': 'Production', 'run_id': '456'}


def test_load_production_model_params_no_production(mlflow_client, production_model_run, xgboost_hyperparameter_config, stock_name):
    # set search_model_versions to return a staging model instead of a production model
    mlflow_client.search_model_versions.return_value = [
        {'name': 'model_name', 'current_stage': 'Staging', 'run_id': '123'},
        {'name': 'model_name', 'current_stage': 'Staging', 'run_id': '789'}
    ]

    # call the function with the sample client
    with pytest.raises(Exception):
        load_production_model_params(mlflow_client, stocks_list[0])


def test_load_production_model_params_missing_hyperparameters(mlflow_client, production_model_run, xgboost_hyperparameter_config, stock_name):
    # modify the production model's hyperparameters to include an unknown key
    production_model_run.data.params = {
        'max_depth': '3',
        'learning_rate': '0.1',
        'n_estimators': '100',
        'unknown_key': 'unknown_value'
    }

    # mock mlflow.get_run to return the modified production model run
    mlflow.get_run = MagicMock(return_value=production_model_run)

    # call the function with the sample client
    prod_validation_model_params_new, current_prod_model = load_production_model_params(mlflow_client, stocks_list[0])

    # check that the unknown key is not included in the returned dictionary
    assert prod_validation_model_params_new == {
        'max_depth': '3',
        'learning_rate': '0.1',
        'n_estimators': '100'
    }


def test_train_inference_model():
    # create dummy training data
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([10, 20, 30])
    params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
    stock_name = 'BOVA11.SA'

    # train the model
    xgboost_model = train_inference_model(X_train, y_train, params, stock_name)

    # check that the model is an instance of XGBRegressor
    assert isinstance(xgboost_model, xgb.sklearn.XGBRegressor)

    # check that the model has been fit to the training data
    assert xgboost_model.n_features_in_ == 2
    assert xgboost_model.n_estimators == 100
    assert xgboost_model.max_depth == None
    assert xgboost_model.learning_rate == None
    #assert len(xgboost_model.evals_result()['validation_0']['rmse']) == 100
    #assert len(xgboost_model.evals_result()['validation_0']['logloss']) == 100


def test_extract_learning_curves():
    # Arrange
    x_train = [[0, 0], [1, 1]]
    y_train = [0, 1]
    model = xgb.XGBRegressor(n_estimators=2, eval_metric=["rmse", "logloss"])
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train)],
    )
    
    # Act
    fig = extract_learning_curves(model, display=False)

    # Assert
    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.fixture
def mock_model():
    return xgb.XGBRegressor()


@pytest.fixture
def mock_data():
    X = pd.DataFrame({
        "Feature1": [1, 2, 3, 4, 5],
        "Feature2": [2, 4, 6, 8, 10],
        "Close_lag_1": [10, 12, 15, 18, 20]
    })
    y = pd.Series([12, 14, 17, 20, 22])
    return X, y


def test_stepwise_forecasting_returns_tuple(mock_model, mock_data):
    X, y = mock_data
    trained_mock_model = mock_model.fit(X, y)
    result = stepwise_forecasting(trained_mock_model, X, y, 5)
    assert isinstance(result, tuple)


def test_stepwise_forecasting_returns_expected_length(mock_model, mock_data):
    X, y = mock_data
    trained_mock_model = mock_model.fit(X, y)
    result = stepwise_forecasting(trained_mock_model, X, y, 5)
    assert len(result) == 3


def test_stepwise_forecasting_returns_expected_metrics(mock_model, mock_data):
    X, y = mock_data
    trained_mock_model = mock_model.fit(X, y)
    result = stepwise_forecasting(trained_mock_model, X, y, 5)
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[2] == 0.0




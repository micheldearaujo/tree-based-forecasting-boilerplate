# -*- coding: utf-8 -*-
# ------------------------
# - Utilities script -
# ------------------------

import sys

sys.path.insert(0,'.')

from src.config import *


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


def predictions_sanity_check(client, run_info, y_train: pd.DataFrame, pred_df: pd.DataFrame, model_mape: float, stage_version: str, stock_name: str):
    """
    Check if the predictions are reliable.
    """

    newest_run_id = run_info.run_id
    newest_run_name = run_info.run_name

    # register the model
    logger.debug("Registering the model...")
    model_details = mlflow.register_model(
        model_uri = f"runs:/{newest_run_id}/{newest_run_name}",
        name = f"{model_config[f'REGISTER_MODEL_NAME_{stage_version}']}_{stock_name}"
    )

    # validate the predictions
    # check if the MAPE is less than 3%
    # check if the predictions have similar variation of historical
    logger.debug("Checking if the metrics and forecasts are valid...")
    
    if (model_mape < 0.03) and (0 < pred_df["Forecast"].std() < y_train.std()*1.5):

        # if so, transit to staging
        client.transition_model_version_stage(
            name=f"{model_config[f'REGISTER_MODEL_NAME_{stage_version}']}_{stock_name}",
            version=model_details.version,
            stage='Staging',
        )

        # return the model details
        logger.debug("The model is valid. Returning the model details...")
        return model_details

    else:
        # if not, discard it
        client.delete_model_version(
            name=f"{model_config[f'REGISTER_MODEL_NAME_{stage_version}']}_{stock_name}",
            version=model_details.version,
        )
        
        # return false to indicate that the model is not valid
        return False


def compare_models(client, model_details, stage_version: str, stock_name: str) -> None:


    # get the metrics of the Production model
    models_versions = []
    #for mv in client.search_model_versions(f"name={model_config[f'REGISTER_MODEL_NAME_{stage_version}']}_{stock_name}"):
    for mv in client.search_model_versions("name='{}_{}'".format(model_config[f'REGISTER_MODEL_NAME_{stage_version}'], stock_name)):
        models_versions.append(dict(mv))

    current_prod_model = [x for x in models_versions if x['current_stage'] == 'Production'][0]

    # Extract the current staging model MAPE
    current_model_mape = mlflow.get_run(current_prod_model['run_id']).data.metrics[model_config['VALIDATION_METRIC']]

    # Get the new model MAPE
    candidate_model_mape = mlflow.get_run(model_details.run_id).data.metrics[model_config['VALIDATION_METRIC']]

    # compare the MAPEs
    print('\n')
    print("-"*10 + " Continous Deployment Results " + "-"*10)

    if candidate_model_mape <= current_model_mape:

        print(f"Candidate model has a better or equal {model_config['VALIDATION_METRIC']} than the active model. Switching models...")
        
        # archive the previous version
        client.transition_model_version_stage(
            name=f"{model_config[f'REGISTER_MODEL_NAME_{stage_version}']}_{stock_name}",
            version=current_prod_model['version'],
            stage='Archived',
        )

        # transition the newest version
        client.transition_model_version_stage(
            name=f"{model_config[f'REGISTER_MODEL_NAME_{stage_version}']}_{stock_name}",
            version=model_details.version,
            stage='Production',
        )


    else:
        print(f"Active model has a better {model_config['VALIDATION_METRIC']} than the candidate model.\nTransiting the new staging model to None.")
        
        client.transition_model_version_stage(
            name=f"{model_config[f'REGISTER_MODEL_NAME_{stage_version}']}_{stock_name}",
            version=model_details.version,
            stage='None',
        )
        
    print(f"Candidate {model_config['VALIDATION_METRIC']}: {candidate_model_mape}\nCurrent {model_config['VALIDATION_METRIC']}: {current_model_mape}")
    print("-"*50)
    print('\n')


def cd_pipeline(run_info, y_train: pd.Series, pred_df: pd.DataFrame, model_mape: float, stock_name: str) -> None:

    logger.debug(" ----- Starting CD pipeline -----")
    
    # create a new Mlflow client
    client = MlflowClient()

    # validate the predictions
    model_details = predictions_sanity_check(client, run_info, y_train, pred_df, model_mape, "VAL", stock_name)

    if model_details:
        # compare the new model with the production model
        logger.info("The model is reliable. Comparing it with the production model...")
        compare_models(client, model_details, "VAL", stock_name)

    else:
        logger.info("The model is not reliable. Discarding it.")

    logger.debug(" ----- CD pipeline finished -----")



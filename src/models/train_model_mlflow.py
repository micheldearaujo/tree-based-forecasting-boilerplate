# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *
from xgboost import plot_importance
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger("model-training")
logger.setLevel(logging.DEBUG)
        

def train_model(X_train, y_train, params):
    """Treina um modelo XGBoost para regressão."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain)
    return model

def predict(model, X_test):
    """Realiza previsões com o modelo XGBoost treinado."""
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    return y_pred


def train_inference_model(X_train:pd.DataFrame, y_train: pd.Series, stock_name: str) -> xgb.sklearn.XGBRegressor:
    """
    Trains the XGBoost model with the full dataset to perform out-of-sample inference.
    """
    
    # use existing params
    xgboost_model = xgb.XGBRegressor(
        eval_metric=["rmse", "logloss"],
        n_estimators=40,
        max_depth=11
    )

    # train the model
    xgboost_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=20
    )

    return xgboost_model


def extract_learning_curves(model: xgb.sklearn.XGBRegressor, display: bool=False) -> matplotlib.figure.Figure:
    """
    Extracting the XGBoost Learning Curves.
    Can display the figure or not.

    Args:
        model (xgb.sklearn.XGBRegressor): Fitted XGBoost model
        display (bool, optional): Display the figure. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Learning curves figure
    """

    # extract the learning curves
    learning_results = model.evals_result()

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    plt.suptitle("XGBoost Learning Curves")
    axs[0].plot(learning_results['validation_0']['rmse'], label='Training')
    axs[0].set_title("RMSE Metric")
    axs[0].set_ylabel("RMSE")
    axs[0].set_xlabel("Iterations")
    axs[0].legend()

    axs[1].plot(learning_results['validation_0']['logloss'], label='Training')
    axs[1].set_title("Logloss Metric")
    axs[1].set_ylabel("Logloss")
    axs[1].set_xlabel("Iterations")
    axs[1].legend()

    fig2, axs2 = plt.subplots(figsize=(6, 3))
    plot_importance(model, ax=axs2, importance_type='gain')
    plt.close()
    

    if display:
        plt.show()
        
    
    return fig, fig2


def train_pipeline():

    client = MlflowClient()
    
    logger.debug("Loading the featurized dataset..")
    stock_df_feat_all = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])

    for stock_name in stock_df_feat_all["Stock"].unique():

        stock_df_feat = stock_df_feat_all[stock_df_feat_all["Stock"] == stock_name].drop("Stock", axis=1).copy()

        logger.info("Creating training dataset for stock %s..."%stock_name)
        X_train=stock_df_feat.drop([model_config["TARGET_NAME"], "Date"], axis=1)
        y_train=stock_df_feat[model_config["TARGET_NAME"]]

        mlflow.set_experiment(experiment_name="Training_Inference_Models")
        with mlflow.start_run(run_name=f"model_inference_{stock_name}") as run:

            logger.debug("Training the model..")
            xgboost_model = train_inference_model(X_train, y_train, stock_name)

            logger.debug("Plotting the learning curves..")
            learning_curves_fig , feat_importance_fig = extract_learning_curves(xgboost_model)

            logger.debug("Logging the results..")
            mlflow.log_params(xgboost_model.get_xgb_params())
            mlflow.log_figure(learning_curves_fig, f"learning_curves_{stock_name}.png")

            logger.debug(f"Logging the model to MLflow...")
            model_signature = infer_signature(X_train, pd.DataFrame(y_train))
            mlflow.xgboost.log_model(
                xgb_model=xgboost_model,
                artifact_path=f"{model_config['MODEL_NAME']}_{stock_name}",
                input_example=X_train.head(),
                signature=model_signature
            )
            logger.debug(f"artifact path: {model_config['MODEL_NAME']}_{stock_name}")

            # register the model
            logger.debug("Registering the model to MLflow...")
            logger.debug(f"Model registration URI: runs:/{run.info.run_id}/{run.info.run_name}")
            logger.debug(f"run_name: model_inference_{stock_name}")
            model_details = mlflow.register_model(
                model_uri = f"runs:/{run.info.run_id}/{run.info.run_name}",
                name = f"{model_config[f'REGISTER_MODEL_NAME_INF']}_{stock_name}"
            )
            
            logger.info(f"Transitioning model versions...")
            models_versions = []

            for mv in client.search_model_versions("name='{}_{}'".format(model_config[f'REGISTER_MODEL_NAME_INF'], stock_name)):
                models_versions.append(dict(mv))

            # Check if there is production model
            try:
                logger.debug("Previous model version found. Transitioning the old version to Archived...") 

                current_prod_inf_model = [x for x in models_versions if x['current_stage'] == 'Production'][0]

                # Archive the previous version
                client.transition_model_version_stage(
                    name=f"{model_config['REGISTER_MODEL_NAME_INF']}_{stock_name}",
                    version=current_prod_inf_model['version'],
                    stage='Archived',
                )

                # transition the newest version
                client.transition_model_version_stage(
                    name=f"{model_config['REGISTER_MODEL_NAME_INF']}_{stock_name}",
                    version=model_details.version,
                    stage='Production',
                )
                
                logger.debug("Successfuly setup the new version as Production!") 

            except IndexError:

                # just set the new model as production
                logger.debug("No Previous model version found. Transitioning this version to Archived...") 

                client.transition_model_version_stage(
                    name=f"{model_config['REGISTER_MODEL_NAME_INF']}_{stock_name}",
                    version=model_details.version,
                    stage='Production',
                )


# Execute the whole pipeline
if __name__ == "__main__":
    logger.info("Starting the training pipeline...")

    train_pipeline()

    logger.info("Training Pipeline was sucessful!\n")


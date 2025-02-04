
def get_dbutils(spark):
    try:
        from pyspark.dbutils import DBUtils

        dbutils = DBUtils(spark)
    except ImportError:
        import IPython

        dbutils = IPython.get_ipython().user_ns["dbutils"]
    return dbutils


def check_and_delete_dir(dir_path):
    try:
        dbutils.fs.ls(dir_path)
        dbutils.fs.rm(dir_path, recurse=True)
        print(f"Deleted directory since it exists: {dir_path}")
    except Exception:
        print(f"No need to delete directory since it does not exist: {dir_path}")


def write_to_datalake_csv(df, fileprefix):
    """
    Write Spark datframe to datalake as csv file

    :param df: spark dataframe ,file_prefix
    :return: na
    """

    filepath = fileprefix + ".dir/"
    df.repartition(1).write.mode("overwrite").format("com.databricks.spark.csv").option(
        "header", "true"
    ).option("delimiter", ",").option("multiLine", "true").option("quote", '"').option(
        "escape", '"'
    ).save(
        filepath
    )

    dbutils = get_dbutils(spark)
    listFiles = dbutils.fs.ls(filepath)

    for subFiles in listFiles:
        if subFiles.name[-4:] == ".csv":
            dbutils.fs.cp(filepath + subFiles.name, fileprefix + ".csv")

    dbutils.fs.rm(filepath, recurse=True)
    print("File has been written to datalake")


# COMMAND ----------


def simple_wape(y_true, y_pred):
    """Calculates simple wape"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.round(
        abs(y_true - y_pred).sum() / abs(y_true).sum()
        if abs(y_true).sum() != 0
        else np.inf,
        5,
    )


# COMMAND ----------


def mean_percent_error(y_true, y_pred):
    """
    Calculate the Mean Percent Error metric
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.round(
        ((y_true - y_pred).sum() / y_true.sum()) if abs(y_true).sum() != 0 else np.inf,
        7,
    )


# COMMAND ----------


def plot_future_forecast(
    training_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    serie: str,
    forecast_horizon: int,
    model_type: str,
    y_title: str = "Tires (millions)",
):
    """
    Plots the historical values and the out-of-sample forecasting, with the confidence interval.

    Args:
        training_df (pd.DataFrame): The training dataframe, with full historical values.
        predictions_df (pd.DataFrame): The predictions dataframe, with the forecasted values.
        serie (str): The name of the serie (Only to display the in title).
        forecast_horizon (int): The amount of months predicted into the future (Only to display in the title).

    Returns:
        None
    """

    # Plot the results
    fig, axs = plt.subplots(figsize=(18, 5))

    # plot the actuals
    sns.lineplot(data=training_df, x="ds", y="y", ax=axs, label="Actuals")
    sns.scatterplot(
        data=training_df,
        x="ds",
        y="y",
        ax=axs,
        size="y",
        sizes=(80, 80),
        legend=False,
    )

    # plot the forecast
    sns.lineplot(data=predictions_df, x="ds", y="yhat", ax=axs, label="Forecast")
    sns.scatterplot(
        data=predictions_df,
        x="ds",
        y="yhat",
        ax=axs,
        size="yhat",
        sizes=(80, 80),
        legend=False,
    )

    axs.fill_between(
        predictions_df.ds,
        predictions_df["lower_end"],
        predictions_df["higher_end"],
        alpha=0.2,
        color="blue",
    )

    axs.set_title(f"{model_type} {forecast_horizon} months Forecast - {serie}")
    axs.set_xlabel("Date")
    axs.set_ylabel(y_title)
    plt.show()


# COMMAND ----------


def forecast_features_prophet(
    past_dataset: pd.DataFrame, future_dataset_clean: pd.DataFrame
) -> pd.DataFrame:
    """
    This function performs the data manipulation to train the model on the past values + lag values
    in order to predict only the necessary values for the feature.
    For example, our forecast horizon is 12 monhts, and the feature is the lag 10
    so we need to train the model until the 10th month and only predict the last 2 months.

    Steps:
    1 - Concatenate the past past_dataset, which contain the data until the current month, with the future dataset,
    which contains the future values (the lags of the actuals). Then we will have the full table to train the model
    Extract the forecast horizon based on the amount o nans
    2 - Train the model until the last existing value
    3 - Perform the forecast for the amount of months that are missing
    4 - Add those predictions to the future dataset
    5 - Return the future dataset with all future months filled

    Parameters:
        past_dataset (pandas dataframe): The historical dataframe, with traning data
        future_dataset_clean (pandas dataframe): The future-indexed dataframe for the forecast

    Returns:
        future_dataset (with the Features to perform prediction), features (a list of features)
    """

    # -- Step 1
    future_dataset = future_dataset_clean.copy()
    past_dataset = past_dataset.set_index("ds")
    model_dataset = pd.concat([past_dataset, future_dataset], axis=0)
    features = []

    for feature in model_dataset.drop(["y", "Key"], axis=1).columns:
        if feature == "covid":
            pass
        else:
            FEAT_FORECAST_HORIZON = sum(model_dataset[[feature]][feature].isna())
            features.append(feature)

            # Verify if there is a lag 12 - no need to predict
            if FEAT_FORECAST_HORIZON != 0:
                feature_dataset = model_dataset[[feature]].dropna(subset=[feature])

                # -- Step 2
                # Train the ETS on the whole dataset
                model_ets = ets.ExponentialSmoothing(
                    feature_dataset,
                    trend="add",
                    damped_trend=True,
                    seasonal="add",
                    seasonal_periods=12,
                ).fit()
                # model_ets = train_ets(feature_dataset)

                # Perform the test
                # -- Step 3
                predictions_df = pd.DataFrame(
                    model_ets.forecast(steps=FEAT_FORECAST_HORIZON), columns=[feature]
                )

                # -- Step 4
                future_dataset.loc[predictions_df.index, feature] = predictions_df[
                    feature
                ]

            else:
                pass

    future_dataset = future_dataset.reset_index()
    future_dataset = future_dataset.rename(columns={"index": "ds"})

    return future_dataset, features


# COMMAND ----------


def create_empty_future_dataframe(
    complete_dataset: pd.DataFrame, inference_forecast_horizon: int
) -> pd.DataFrame:
    """
    Creates the empty dataframe with the future timestamp index in order to make the out-of-sample forecasting.

    Args:
        training_df (pd.DataFrame): The training dataframe, with full historical values.

    Returns:
        pd.DataFrame: The DataFrame with the future dates in the index.
    """

    # make it for Prophet
    try:
        future_dates_list = [
            complete_dataset.ds.max() + dateutil.relativedelta.relativedelta(months=i)
            for i in range(1, inference_forecast_horizon + 1)
        ]
    # make it for ARIMA
    except AttributeError:
        future_dates_list = [
            complete_dataset.index.max()
            + dateutil.relativedelta.relativedelta(months=i)
            for i in range(1, inference_forecast_horizon + 1)
        ]

    future_df = pd.DataFrame(index=future_dates_list)

    return future_df


# COMMAND ----------


def sort_features(training_df: pd.DataFrame) -> tuple:
    """
    Sort the list of features so we always keep the columns in the same order.

    Args:
        training_df (pd.DataFrame): The training dataframe, with full historical values.

    Returns:
        tuple: A tuple with 2 elements:
            1. training_df (pd.DataFrame): The inputed training dataframe, with the columns reordered.
            2. sorted_features (list): A list with the sorted features, without the non-feature columns (y, key, ds)
    """

    # Sort the columns
    features = list(training_df.columns)
    sorted_features = sorted(features)
    training_df = training_df.reindex(columns=sorted_features)

    # - Now remove the SALES & key columns out of the features list
    try:
        features = list(training_df.drop(["y", "Key", "ds"], axis=1).columns)
    except KeyError:
        features = list(training_df.drop([Y_LABEL, "Key"], axis=1).columns)
    sorted_features = sorted(features)

    return (training_df, sorted_features)


# COMMAND ----------


def train_snaive(train_series, seasonal_periods, forecast_horizon):
    """
    Python implementation of Seasonal Naive Forecast.
    This should work similar to https://otexts.com/fpp2/simple-methods.html
    Returns two arrays
     > fitted: Values fitted to the training dataset
     > fcast: seasonal naive forecast

    Author: Sandeep Pawar

    Date: Apr 9, 2020

    Ver: 1.0

    train_series: Pandas Series
        Training Series to be used for forecasting. This should be a valid Pandas Series.
        Length of the Training set should be greater than or equal to number of seasonal periods

    Seasonal_periods: int
        No of seasonal periods
        Yearly=1
        Quarterly=4
        Monthly=12
        Weekly=52

    Forecast_horizon: int
        Number of values to forecast into the future

    e.g.
    fitted_values = pysnaive(train,12,12)[0]
    fcast_values = pysnaive(train,12,12)[1]
    """

    if (
        len(train_series) >= seasonal_periods
    ):  # checking if there are enough observations in the training data

        last_season = train_series.iloc[-seasonal_periods:]

        reps = np.int(np.ceil(forecast_horizon / seasonal_periods))

        fcarray = np.tile(last_season, reps)

        fcast = pd.Series(fcarray[:forecast_horizon])

        fitted = train_series.shift(seasonal_periods)

    else:
        fcast = print(
            "Length of the trainining set must be greater than number of seasonal periods"
        )

    return fitted, fcast


# COMMAND ----------


def configure_mlflow_experiment(serie, model_type, experiment_path):
    """
    Creates the MLFlow experiment within Databricks Workflows.

    Args:
        serie (str): The name of the serie granularity.
        model_type (str): MLFlow registered model name.

    Returns:
        None
    """

    if mlflow.get_experiment_by_name(f"{experiment_path}-{serie}-{model_type}") is None:
        experiment_id = mlflow.create_experiment(
            f"{experiment_path}-{serie}-{model_type}"
        )
    else:
        experiment_id = mlflow.get_experiment_by_name(
            f"{experiment_path}-{serie}-{model_type}"
        ).experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)


# COMMAND ----------


def log_everything_to_mlflow(
    model: prophet.forecaster.Prophet,
    training_df: pd.DataFrame,
    testing_df: pd.DataFrame,
    serie: str,
    model_type: str,
    metrics: list,
    fig: matplotlib.figure.Figure,
) -> None:
    """
    Helper to log the MLflow items: Model, metrics, parameters and artifacts.

    Args:
        model (prophet.forecaster.Prophet): Trained Model.
        training_df (pd.DataFrame): Training DataFrame.
        testing_df (pd.DataFrame): Testing DataFrame.
        serie (str): The name of the serie granularity.
        model_type (str): MLFlow registered model name.
        metrics (list): List with all the model metrics.
        fig (matplotlib.figure.Figure): Figure of the testing results.

    Returns:
        None
    """

    log_model(model, training_df, testing_df, serie, model_type)
    log_all_metrics(metrics)
    log_all_params(model)
    log_all_artifacts(fig, serie)


# COMMAND ----------


def log_model(
    model: prophet.forecaster.Prophet,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    serie: str,
    model_type: str,
) -> None:
    """
    Performs the MLFlow model logging.

    Args:
        model (prophet.forecaster.Prophet): The FB Prophet model trained on the validation dataset.
        train_df (pd.DataFrame): The Training DataFrame.
        test_df (pd.DataFrame): The Testing DataFrame.
        serie (str): The name of the Serie.
        model_type (str): The name of the model.

    Returns:
        None
    """

    if "prophet" in model_type.lower():

        model_signature = infer_signature(
            model_input=train_df.drop("y", axis=1),
            model_output=pd.DataFrame(train_df["y"]),
        )

        mlflow.prophet.log_model(
            pr_model=model,
            artifact_path=f"{serie}-{model_type}",
            signature=model_signature,
            input_example=test_df.head(5),
        )

    elif ("sarima" in model_type.lower()) or ("arima" in model_type.lower()):

        model_signature = infer_signature(
            model_input=train_df.drop(Y_LABEL, axis=1),
            model_output=pd.DataFrame(train_df[Y_LABEL]),
        )

        mlflow.pmdarima.log_model(
            pmdarima_model=model,
            artifact_path=f"{serie}-{model_type}",
            signature=model_signature,
            input_example=test_df.head(5),
        )


def log_all_metrics(metrics_list: list) -> None:
    """
    Performs the MLFlow validation metrics logging.

    Args:
        metrics_list (list): The list containing all the validation metrics.

    Returns:
        None
    """

    # log the metrics
    mlflow.log_metric("MAPE", metrics_list[0])
    mlflow.log_metric("WAPE", metrics_list[6])
    mlflow.log_metric("MAE", metrics_list[2])


def log_all_artifacts(fig: matplotlib.figure.Figure, serie: str) -> None:
    """
    Performs the MLFlow validation metrics logging.

    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure of the validation lineplot.
        serie (str): The name of the Serie.

    Returns:
        None
    """
    mlflow.log_figure(fig, f"{serie}-prophet_predictions.png")


def log_all_params(model: prophet.forecaster.Prophet) -> None:
    """
    Performs the MLFlow parameters logging.

    Args:
        model (prophet.forecaster.Prophet): The FB Prophet model trained on the validation dataset.

    Returns:
        None
    """

    if type(model) == prophet.forecaster.Prophet:
        mlflow.log_param("mcmc_samples", model.mcmc_samples)
        mlflow.log_param("interval_width", model.interval_width)
        mlflow.log_param("growth", model.growth)
        mlflow.log_param("weekly_seasonality", model.weekly_seasonality)
        mlflow.log_param("daily_seasonality", model.daily_seasonality)
        mlflow.log_param("yearly_seasonality", model.yearly_seasonality)
        mlflow.log_param("changepoints", model.changepoints)
        mlflow.log_param("n_changepoints", model.n_changepoints)
        mlflow.log_param("uncertainty_samples", model.uncertainty_samples)
        mlflow.log_param("seasonalities", model.seasonalities)
        mlflow.log_param("logistic_floor", model.logistic_floor)

    elif type(model) == pmdarima.arima.arima.ARIMA:
        mlflow.log_param("Order", str(model.order))
        mlflow.log_param("Seasonal Order", str(model.seasonal_order))
        mlflow.log_param("Method", model.method)

    else:
        print("Unreconized model!!")


def register_model(
    model: prophet.forecaster.Prophet,
    model_uri: str,
    model_name: str,
    stage_name: str = "Staging",
):
    """
    Performs the model registry using MLFlow and set the run as determinated stage defined by the stage_name variable.

    Args:
         model (prophet.forecaster.Prophet): The FB Prophet model trained on the validation dataset.
         model_uri (str): The link to the MLFlow model run.
         model_name (str): The MLFlow registered Model.
         stage_name (str): The staging name to set the model ("Staging" or "Production").

    Returns:
        None
    """
    # register the model
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

    # set to staging for now
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage=stage_name,
    )


# COMMAND ----------


def check_feature_significance(coef_lower: int, coef_upper: int) -> str:
    """
    Checks if a given regressor (variable) was significant or not to the model, by looking at the regressor coeficient confidence interval.
    OBS: This function is meant to be used inside pandas.apply() method.

    Args:
        coef_lower (int): The lower boundary of the coeficient distribution
        coef_upper (int): The upper boundary of the coeficient distribution

    Returns:
        str: String telling if the factor is significant or not
    """

    if coef_lower < 0 < coef_upper:
        return "not significant"

    else:
        return "significant"


# COMMAND ----------


def plot_train_test_result(
    training_df: pd.DataFrame,
    testing_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    serie: str,
    model_wape: float,
    model_mape: float,
    true_label: str = "y",
    predicted_label: str = "yhat",
    prediction_name: str = "Prophet Multivariate",
    y_title: str = "Tires (millions)",
):

    fig, axs = plt.subplots(figsize=(18, 4))
    training_df = pd.DataFrame(training_df)
    predictions_df = pd.DataFrame(predictions_df)

    training_df = training_df.set_index("ds")
    testing_df = testing_df.set_index("ds")
    predictions_df = predictions_df.set_index("ds")

    sns.lineplot(
        data=training_df,
        x=training_df.index,
        y=true_label,
        ax=axs,
        label="Training values",
    )

    sns.scatterplot(
        data=training_df,
        x=training_df.index,
        y=true_label,
        ax=axs,
        size=true_label,
        sizes=(80, 80),
        legend=False,
    )

    sns.lineplot(
        data=testing_df,
        x=testing_df.index,
        y=true_label,
        ax=axs,
        label="Testing values",
    )

    sns.scatterplot(
        data=testing_df,
        x=testing_df.index,
        y=true_label,
        ax=axs,
        size=true_label,
        sizes=(80, 80),
        legend=False,
    )

    sns.lineplot(
        data=predictions_df,
        x=predictions_df.index,
        y=predicted_label,
        ax=axs,
        label=prediction_name,
    )

    sns.scatterplot(
        data=predictions_df,
        x=predictions_df.index,
        y=predicted_label,
        ax=axs,
        size=predicted_label,
        sizes=(80, 80),
        legend=False,
    )

    # plot the confidence interval
    axs.fill_between(
        predictions_df.index,
        predictions_df["lower_end"],
        predictions_df["higher_end"],
        alpha=0.2,
        color="blue",
    )

    axs.set_title(
        f"{prediction_name} Forecast - {serie}\n WAPE = {round(model_wape*100, 2)}% | MAPE = {round(model_mape*100, 2)}%"
    )

    axs.set_xlabel("Date")
    axs.set_ylabel(y_title)
    # plt.savefig(f"/tmp/{serie}-predictions.png")
    plt.show()

    return fig, axs


# COMMAND ----------


def plot_train_test_result_arima(
    dataset,
    serie,
    test_df,
    predictions_df,
    true_label,
    predicted_label,
    model_wape,
    model_mape,
    prediction_name,
):

    fig, axs = plt.subplots(figsize=(18, 4))
    dataset = pd.DataFrame(dataset)
    test_df = pd.DataFrame(test_df)

    sns.lineplot(
        data=dataset, x=dataset.index, y=true_label, ax=axs, label="Training values"
    )

    sns.scatterplot(
        data=dataset,
        x=dataset.index,
        y=true_label,
        ax=axs,
        size=true_label,
        sizes=(80, 80),
        legend=False,
    )

    sns.lineplot(
        data=test_df, x=test_df.index, y=true_label, ax=axs, label="Testing values"
    )

    sns.scatterplot(
        data=test_df,
        x=test_df.index,
        y=true_label,
        ax=axs,
        size=true_label,
        sizes=(80, 80),
        legend=False,
    )

    sns.lineplot(
        data=predictions_df,
        x=predictions_df.index,
        y=predicted_label,
        ax=axs,
        label=prediction_name,
    )

    sns.scatterplot(
        data=predictions_df,
        x=predictions_df.index,
        y=predicted_label,
        ax=axs,
        size=predicted_label,
        sizes=(80, 80),
        legend=False,
    )

    axs.set_title(
        f"{prediction_name} Forecast - {serie}\n WAPE = {round(model_wape*100, 2)}% | MAPE = {round(model_mape*100, 2)}%"
    )
    axs.set_xlabel("Date")
    axs.set_ylabel("Tires")
    # plt.legend('best')
    plt.savefig(f"/tmp/{serie}-predictions.png")

    return fig, axs


# COMMAND ----------


def plot_train_test_result_xgboost(
    dataset,
    serie,
    test_df,
    predictions_df,
    true_label,
    predicted_label,
    model_rmse,
    model_mape,
    prediction_name,
):
    # Plot the results

    return fig, axs


# COMMAND ----------


def walk_forward_validation_baseline(
    dataset: pd.DataFrame,
    forecast_horizon: int,
    sliding_window: int,
    min_train_size: int,
    step_size: int,
    plot=False,
):
    """
    This function implements a walk-forward validation on one time series
    It acts like a train_test_split, getting the full dataset and splitting it
    based on the step_size and prediction_window.

    :param dataset: The complete dataset of each serie
    :param min_train_size: The minimun amount of days to train the model
    :param forecast_horizon: The amount of periods to predict into the future
    :param step_size: The amount of days to walk forward

    returns: The overall average of the scores
    """

    # Getting the number of iterations
    number_of_steps = (dataset.shape[0] - min_train_size - step_size) // step_size

    # Define the first train size
    mapes = []
    rmses = []
    current_train_size = min_train_size - step_size

    print(f"The number of steps is: {number_of_steps}")
    for step in range(number_of_steps):

        # First create the base training dataset
        train = dataset.iloc[: current_train_size + step_size]
        print(f"Training size: {train.shape[0]}")

        # Create the test set
        test = dataset.iloc[
            current_train_size
            + step_size : current_train_size
            + step_size
            + forecast_horizon
        ]
        predictions = test.copy()
        print(f"test size: {test.shape[0]}")

        # update the train_size
        current_train_size += step_size

        # Now we call the model fuctions, in this case, the base line

        # Fitted values
        py_snaive_fit = train_snaive(
            train["SALES"], seasonal_periods=12, forecast_horizon=forecast_horizon
        )[0]

        # forecast
        py_snaive = train_snaive(
            train["SALES"], seasonal_periods=12, forecast_horizon=forecast_horizon
        )[1]

        predictions["snaive_predictions"] = py_snaive.values

        # calculate the metrics
        model_r2 = r2_score(predictions["SALES"], predictions["snaive_predictions"])
        model_mape = np.sqrt(
            mean_absolute_percentage_error(
                predictions["SALES"], predictions["snaive_predictions"]
            )
        )
        model_rmse = np.sqrt(
            mean_squared_error(predictions["SALES"], predictions["snaive_predictions"])
        )
        model_mae = mean_absolute_error(
            predictions["SALES"], predictions["snaive_predictions"]
        )
        forecast_bias = (
            (predictions["snaive_predictions"].sum() / predictions["SALES"].sum()) - 1
        ) * 100
        mapes.append(round(model_mape, 2))
        rmses.append(round(model_rmse, 2))

        if plot:
            plot_train_test_result(
                train,
                test,
                predictions,
                "SALES",
                "snaive_predictions",
                model_rmse,
                model_mape,
            )

    return mapes, rmses


# COMMAND ----------


def ts_train_test_split_outofsample(dataset: pd.DataFrame, forecast_horizon: int):
    """
    This function acts like a regular_train_test_split but it doest not shuffle the dataset
    because it needs to be ordered in time.
    It splits the dataset in some test_size percentage, where the first part is the training
    and the last is the testing.

    :param dataset: The dataset that you want to split
    :param forecast_horizon: The amount of periods to forecast

    retuns: the train, test dataframes and the index of start and end of the test.
    """

    # Spliting into training and testing.
    train = dataset
    test = dataset.iloc[len(dataset) - forecast_horizon :]

    # Defining the index number of the start and end of the testing Window
    start_test = len(train)
    end_test = len(train) + len(test) - 1

    return train, test, start_test, end_test


# COMMAND ----------


def ts_train_test_split(dataset: pd.DataFrame, forecast_horizon: int):
    """
    This function acts like a regular_train_test_split but it doest not shuffle the dataset
    because it needs to be ordered in time.
    It splits the dataset in some test_size percentage, where the first part is the training
    and the last is the testing.

    :param dataset: The dataset that you want to split
    :param forecast_horizon: The amount of periods to forecast

    retuns: the train, test dataframes and the index of start and end of the test.
    """

    # Spliting into training and testing.
    train = dataset.iloc[: len(dataset) - forecast_horizon]
    test = dataset.iloc[len(dataset) - forecast_horizon :]

    # Defining the index number of the start and end of the testing Window
    start_test = len(train)
    end_test = len(train) + len(test) - 1

    return train, test, start_test, end_test


# COMMAND ----------


def train_auto_arima_univariate(
    y,
    forecast_horizon,
    with_intercept=False,
    start_p=0,
    d=1,
    start_q=0,
    max_p=12,
    max_d=5,
    max_q=12,
    start_P=0,
    max_P=12,
    D=1,
    start_Q=0,
    max_D=5,
    max_Q=12,
    m=12,
    seasonal=True,
    error_action="warn",
    trace=False,
    supress_warnings=True,
    stepwise=True,
    random_state=42,
    n_fits=300,
    max_iter=300,
    n_jobs=4,
):

    model = auto_arima(
        y,
        start_p=start_p,
        d=d,
        start_q=start_q,
        max_p=max_p,
        max_d=max_d,
        max_q=max_q,
        start_P=start_P,
        max_P=max_P,
        D=D,
        start_Q=start_Q,
        max_D=max_D,
        max_Q=max_Q,
        m=m,
        seasonal=seasonal,
        error_action=error_action,
        trace=trace,
        supress_warnings=supress_warnings,
        stepwise=stepwise,
        random_state=random_state,
        n_fits=n_fits,
        scoring="mae",
        with_intercept=with_intercept,
        out_of_sample_size=forecast_horizon,
    )

    return model


# COMMAND ----------


def train_auto_arima_multivariate(
    y,
    x,
    forecast_horizon,
    with_intercept=False,
    start_p=0,
    d=1,
    start_q=0,
    max_p=4,
    max_d=2,
    max_q=4,
    start_P=0,
    max_P=12,
    D=1,
    start_Q=0,
    max_D=2,
    max_Q=12,
    m=12,
    seasonal=True,
    error_action="warn",
    trace=False,
    supress_warnings=True,
    stepwise=True,
    random_state=42,
    n_fits=300,
    max_iter=300,
    n_jobs=4,
):

    model = auto_arima(
        y,
        x,
        start_p=start_p,
        d=d,
        start_q=start_q,
        max_p=max_p,
        max_d=max_d,
        max_q=max_q,
        start_P=start_P,
        max_P=max_P,
        D=D,
        start_Q=start_Q,
        max_D=max_D,
        max_Q=max_Q,
        m=m,
        seasonal=seasonal,
        error_action=error_action,
        trace=trace,
        supress_warnings=supress_warnings,
        stepwise=stepwise,
        random_state=random_state,
        n_fits=n_fits,
        scoring="mae",
        with_intercept=with_intercept,
        out_of_sample_size=forecast_horizon,
    )

    return model


# COMMAND ----------


def test_modelX_scaled(
    model, test, test_scaled, true_label, predicted_label, model_name, target_scaler
):

    if model_name == "arima":

        predictions_array, conf_int = model.predict(
            X=test_scaled.drop(Y_LABEL, axis=1).values,
            n_periods=FORECAST_HORIZON,
            return_conf_int=True,
            alpha=0.05,
        )

        predictions = pd.DataFrame(predictions_array, index=test_scaled.index)
        predictions.columns = [PREDICTED_LABEL]
        predictions[PREDICTED_LABEL] = target_scaler.inverse_transform(
            predictions[[PREDICTED_LABEL]]
        )
        # predictions[true_label] = test[true_label]
        # test[PREDICTED_LABEL] = predictions[PREDICTED_LABEL]
        # feat_to_transform = test.drop("SALES", axis=1).columns
        # test[feat_to_transform] = scaler.inverse_transform(test[feat_to_transform])
        # Scale back the predictions

        # Extract the metrics
        model_r2 = round(r2_score(test[true_label], predictions[predicted_label]), 2)
        model_mape = round(
            mean_absolute_percentage_error(
                test[true_label], predictions[predicted_label]
            ),
            2,
        )
        model_rmse = round(
            np.sqrt(mean_squared_error(test[true_label], predictions[predicted_label])),
            2,
        )
        model_mae = round(
            mean_absolute_error(test[true_label], predictions[predicted_label]), 2
        )
        forecast_bias = round(
            ((predictions[predicted_label].sum() / test[true_label].sum()) - 1), 2
        )
        model_mpe = round(john_metric(test[true_label], predictions[predicted_label]))
        # print(conf_int)

        return (
            predictions,
            [model_mape, model_rmse, model_mae, model_r2, forecast_bias, model_mpe],
            conf_int,
        )

    elif model_name == "prophet":

        future_pd = model.make_future_dataframe(
            periods=FORECAST_HORIZON, freq="m", include_history=False
        )

        # predict over the dataset
        predictions = model.predict(test)

        # Extract the metrics
        model_r2 = round(r2_score(test[true_label], predictions[predicted_label]), 2)
        model_mape = round(
            mean_absolute_percentage_error(
                test[true_label], predictions[predicted_label]
            ),
            2,
        )
        model_rmse = round(
            np.sqrt(mean_squared_error(test[true_label], predictions[predicted_label])),
            2,
        )
        model_mae = round(
            mean_absolute_error(test[true_label], predictions[predicted_label]), 2
        )
        forecast_bias = round(
            ((predictions[predicted_label].sum() / test[true_label].sum()) - 1), 2
        )
        model_mpe = round(john_metric(test[true_label], predictions[predicted_label]))
        # print(conf_int)

        return predictions, [
            model_mape,
            model_rmse,
            model_mae,
            model_r2,
            forecast_bias,
            model_mpe,
        ]


# COMMAND ----------


def test_modelX_scaled1(
    model, test, test_scaled, true_label, predicted_label, model_name, target_scaler
):

    if model_name == "arima":

        predictions_array, conf_int = model.predict(
            X=test_scaled.drop(Y_LABEL, axis=1).values,
            n_periods=FORECAST_HORIZON,
            return_conf_int=True,
            alpha=0.05,
        )

        predictions = pd.DataFrame(predictions_array, index=test_scaled.index)
        predictions.columns = [PREDICTED_LABEL]
        # predictions[predictions.columns] = target_scaler.inverse_transform(predictions)
        # predictions[true_label] = test[true_label]
        # test[PREDICTED_LABEL] = predictions[PREDICTED_LABEL]
        # feat_to_transform = test.drop("SALES", axis=1).columns
        # test[feat_to_transform] = scaler.inverse_transform(test[feat_to_transform])
        # Scale back the predictions

        # Extract the metrics
        model_r2 = round(r2_score(test[true_label], predictions[predicted_label]), 2)
        model_mape = round(
            mean_absolute_percentage_error(
                test[true_label], predictions[predicted_label]
            ),
            2,
        )
        model_rmse = round(
            np.sqrt(mean_squared_error(test[true_label], predictions[predicted_label])),
            2,
        )
        model_mae = round(
            mean_absolute_error(test[true_label], predictions[predicted_label]), 2
        )
        forecast_bias = round(
            ((predictions[predicted_label].sum() / test[true_label].sum()) - 1), 2
        )
        model_mpe = round(john_metric(test[true_label], predictions[predicted_label]))
        # print(conf_int)

        return (
            predictions,
            [model_mape, model_rmse, model_mae, model_r2, forecast_bias, model_mpe],
            conf_int,
        )

    elif model_name == "prophet":

        future_pd = model.make_future_dataframe(
            periods=FORECAST_HORIZON, freq="m", include_history=False
        )

        # predict over the dataset
        predictions = model.predict(test)

        # Extract the metrics
        model_r2 = round(r2_score(test[true_label], predictions[predicted_label]), 2)
        model_mape = round(
            mean_absolute_percentage_error(
                test[true_label], predictions[predicted_label]
            ),
            2,
        )
        model_rmse = round(
            np.sqrt(mean_squared_error(test[true_label], predictions[predicted_label])),
            2,
        )
        model_mae = round(
            mean_absolute_error(test[true_label], predictions[predicted_label]), 2
        )
        forecast_bias = round(
            ((predictions[predicted_label].sum() / test[true_label].sum()) - 1), 2
        )
        model_mpe = round(john_metric(test[true_label], predictions[predicted_label]))
        # print(conf_int)

        return predictions, [
            model_mape,
            model_rmse,
            model_mae,
            model_r2,
            forecast_bias,
            model_mpe,
        ]


# COMMAND ----------


def test_modelX(model, test, true_label, predicted_label, model_name):

    if model_name == "arima":

        predictions_array, conf_int = model.predict(
            X=test.drop(Y_LABEL, axis=1).values,
            n_periods=FORECAST_HORIZON,
            return_conf_int=True,
            alpha=0.05,
        )

        predictions = pd.DataFrame(predictions_array, index=test.index)
        predictions.columns = [PREDICTED_LABEL]
        predictions[true_label] = test[true_label]

        # Extract the metrics
        model_r2 = round(r2_score(test[true_label], predictions[predicted_label]), 2)
        model_mape = round(
            mean_absolute_percentage_error(
                test[true_label], predictions[predicted_label]
            ),
            2,
        )
        model_rmse = round(
            np.sqrt(mean_squared_error(test[true_label], predictions[predicted_label])),
            2,
        )
        model_mae = round(
            mean_absolute_error(test[true_label], predictions[predicted_label]), 2
        )
        forecast_bias = round(
            ((predictions[predicted_label].sum() / test[true_label].sum()) - 1), 2
        )
        model_mpe = round(john_metric(test[true_label], predictions[predicted_label]))
        model_wape = round(
            (predictions[true_label] - predictions[predicted_label]).abs().sum()
            / predictions[true_label].sum(),
            2,
        )

        return (
            predictions,
            [
                model_mape,
                model_rmse,
                model_mae,
                model_r2,
                forecast_bias,
                model_mpe,
                model_wape,
            ],
            conf_int,
        )

    elif model_name == "prophet":

        future_pd = model.make_future_dataframe(
            periods=FORECAST_HORIZON, freq="m", include_history=False
        )

        # predict over the dataset
        predictions = model.predict(test)

        # Extract the metrics
        model_r2 = round(r2_score(test[true_label], predictions[predicted_label]), 2)
        model_mape = round(
            mean_absolute_percentage_error(
                test[true_label], predictions[predicted_label]
            ),
            2,
        )
        model_rmse = round(
            np.sqrt(mean_squared_error(test[true_label], predictions[predicted_label])),
            2,
        )
        model_mae = round(
            mean_absolute_error(test[true_label], predictions[predicted_label]), 2
        )
        forecast_bias = round(
            ((predictions[predicted_label].sum() / test[true_label].sum()) - 1), 2
        )
        model_mpe = round(john_metric(test[true_label], predictions[predicted_label]))
        model_wape = round(
            simple_wape(test[true_label], predictions[predicted_label]), 4
        )
        # print(conf_int)

        return predictions, [
            model_mape,
            model_rmse,
            model_mae,
            model_r2,
            forecast_bias,
            model_mpe,
            model_wape,
        ]


# COMMAND ----------


def train_auto_arima(
    train,
    start_p=0,
    d=1,
    start_q=0,
    max_p=12,
    max_d=5,
    max_q=12,
    start_P=0,
    max_P=12,
    D=1,
    start_Q=0,
    max_D=5,
    max_Q=12,
    m=12,
    seasonal=True,
    error_action="warn",
    trace=True,
    supress_warnings=True,
    stepwise=False,
    random_state=42,
    n_fits=1000,
    max_iter=1000,
    n_jobs=8,
):

    model = auto_arima(
        train,
        start_p=start_p,
        d=d,
        start_q=start_q,
        max_p=max_p,
        max_d=max_d,
        max_q=max_q,
        start_P=start_P,
        max_P=max_P,
        D=D,
        start_Q=start_Q,
        max_D=max_D,
        max_Q=max_Q,
        m=m,
        seasonal=seasonal,
        error_action=error_action,
        trace=trace,
        supress_warnings=supress_warnings,
        stepwise=stepwise,
        random_state=random_state,
        n_fits=n_fits,
    )

    return model


# COMMAND ----------


def test_model(model, test, true_label, predicted_label, model_name):

    if model_name == "arima":

        predictions_array, conf_int = model.predict(
            n_periods=FORECAST_HORIZON, return_conf_int=True, alpha=0.05
        )
        predictions = pd.DataFrame(predictions_array, index=test.index)
        predictions.columns = [PREDICTED_LABEL]

    elif model_name == "prophet":

        future_pd = model.make_future_dataframe(
            periods=FORECAST_HORIZON, freq="m", include_history=False
        )

        # predict over the dataset
        predictions = model.predict(future_pd)

    elif model_name == "ets":

        predictions = test.copy()
        predictions[PREDICTED_LABEL] = model.forecast(steps=len(test))

    elif model_name == "boosting":

        test[0][PREDICTED_LABEL] = model.predict(test[0])
        test[0][Y_LABEL] = test[1]
        predictions = test[0].copy()
        test = test[0]

    # Extract the metrics
    try:
        test = pd.DataFrame(test)
        model_r2 = round(r2_score(test[true_label], predictions[predicted_label]), 2)
        model_mape = round(
            mean_absolute_percentage_error(
                test[true_label], predictions[predicted_label]
            ),
            2,
        )
        model_rmse = round(
            np.sqrt(mean_squared_error(test[true_label], predictions[predicted_label])),
            2,
        )
        model_mae = round(
            mean_absolute_error(test[true_label], predictions[predicted_label]), 2
        )
        forecast_bias = round(
            ((predictions[predicted_label].sum() / test[true_label].sum()) - 1), 2
        )
        model_mpe = round(john_metric(test[true_label], predictions[predicted_label]))
    except:
        model_r2 = 0
        model_mape = 0
        model_rmse = 0
        model_mae = 0
        forecast_bias = 0
        model_mpe = 0

    return (
        predictions,
        [model_mape, model_rmse, model_mae, model_r2, forecast_bias, model_mpe],
        conf_int,
    )


# COMMAND ----------


def train_xgboost(X_train, y_train):
    # Instantiate the model with optimal parameters
    model = XGBRegressor(n_estimators=1000)

    # Fit the model
    model = model.fit(X_train, y_train, verbose=True)

    return model


# COMMAND ----------


def train_xgboost_opt(X_train, y_train):
    # Instantiate the model with optimal parameters
    model = XGBRegressor(
        n_estimators=500,
        colsample_bytree=1.0,
        gamma=0.01,
        learning_rate=1.0,
        max_depth=4,
        reg_lambda=100,
        scale_pos_weight=1,
    )

    # Fit the model
    model = model.fit(X_train, y_train, verbose=True)

    return model


# COMMAND ----------


def walk_forward_validation_auto_arima(
    dataset: pd.DataFrame,
    forecast_horizon: int,
    sliding_window: int,
    min_train_size: int,
    step_size: int,
    prediction_name: str,
    model_name,
    plot=False,
):
    """
    This function implements a walk-forward validation on one time series
    It acts like a train_test_split, getting the full dataset and splitting it
    based on the step_size and prediction_window.

    :param dataset: The complete dataset of each serie
    :param min_train_size: The minimun amount of days to train the model
    :param forecast_horizon: The amount of periods to predict into the future
    :param step_size: The amount of days to walk forward

    returns: The overall average of the scores
    """

    # Getting the number of iterations
    number_of_steps = (dataset.shape[0] - min_train_size - step_size) // step_size

    # Define the first train size
    mapes = []
    rmses = []
    mpes = []
    current_train_size = min_train_size - step_size

    print(f"The number of steps is: {number_of_steps}")
    for step in range(number_of_steps):

        # First create the base training dataset
        train = dataset.iloc[: current_train_size + step_size]
        print(f"Training size: {train.shape[0]}")

        # Create the test set
        test = dataset.iloc[
            current_train_size
            + step_size : current_train_size
            + step_size
            + forecast_horizon
        ]

        predictions = test.copy()
        print(f"test size: {test.shape[0]}")

        # update the train_size
        current_train_size += step_size

        # Now we call the model fuctions, in this case, the base line
        if model_name == "arima":
            model = train_auto_arima(train)

        elif model_name == "ets":
            model = train_ets(train)

        # Perform the test
        predictions, metrics = test_model(
            model, test, Y_LABEL, PREDICTED_LABEL, model_name
        )

        if plot:
            plot_train_test_result_arima(
                train,
                serie,
                test,
                predictions,
                Y_LABEL,
                PREDICTED_LABEL,
                metrics[5],
                metrics[0],
                prediction_name,
            )

        mapes.append(metrics[0])
        rmses.append(metrics[1])
        mpes.append(metrics[5])

    return mapes, mpes


# COMMAND ----------


def walk_forward_validation_auto_arimaX(
    dataset: pd.DataFrame,
    forecast_horizon: int,
    sliding_window: int,
    min_train_size: int,
    step_size: int,
    prediction_name: str,
    model_name,
    plot=False,
):
    """
    This function implements a walk-forward validation on one time series
    It acts like a train_test_split, getting the full dataset and splitting it
    based on the step_size and prediction_window.

    :param dataset: The complete dataset of each serie
    :param min_train_size: The minimun amount of days to train the model
    :param forecast_horizon: The amount of periods to predict into the future
    :param step_size: The amount of days to walk forward

    returns: The overall average of the scores
    """

    # Getting the number of iterations
    number_of_steps = (dataset.shape[0] - min_train_size - step_size) // step_size

    # Define the first train size
    mapes = []
    rmses = []
    mpes = []
    current_train_size = min_train_size - step_size

    print(f"The number of steps is: {number_of_steps}")
    for step in range(number_of_steps):

        # First create the base training dataset
        train = dataset.iloc[: current_train_size + step_size]
        print(f"Training size: {train.shape[0]}")

        # Create the test set
        test = dataset.iloc[
            current_train_size
            + step_size : current_train_size
            + step_size
            + forecast_horizon
        ]

        predictions = test.copy()
        print(f"test size: {test.shape[0]}")

        # update the train_size
        current_train_size += step_size

        # Now we call the model fuctions, in this case, the base line
        if model_name == "arima":
            model = train_auto_arimaX(
                y=train[Y_LABEL],
                x=(train.drop(Y_LABEL, axis=1).values),
            )

        elif model_name == "ets":
            model = train_ets(train)

        # Perform the test
        predictions, metrics = test_modelX(
            model, test, Y_LABEL, PREDICTED_LABEL, "arima"
        )

        if plot:
            plot_train_test_result_arima(
                train,
                serie,
                test,
                predictions,
                Y_LABEL,
                PREDICTED_LABEL,
                metrics[5],
                metrics[0],
                prediction_name,
            )

        mapes.append(metrics[0])
        rmses.append(metrics[1])
        mpes.append(metrics[1])

    return mapes, mpes


# COMMAND ----------


def train_ets(training_data):
    """
    Train and Exponential Smoothing model
    """

    ets_model = ets.ExponentialSmoothing(
        training_data,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
    ).fit()

    return ets_model


# COMMAND ----------


def walk_forward_validation_prophet(
    dataset: pd.DataFrame,
    forecast_horizon: int,
    sliding_window: int,
    min_train_size: int,
    step_size: int,
    plot=False,
):
    """
    This function implements a walk-forward validation on one time series
    It acts like a train_test_split, getting the full dataset and splitting it
    based on the step_size and prediction_window.

    :param dataset: The complete dataset of each serie
    :param min_train_size: The minimun amount of days to train the model
    :param forecast_horizon: The amount of periods to predict into the future
    :param step_size: The amount of days to walk forward

    returns: The overall average of the scores
    """

    # Getting the number of iterations
    number_of_steps = (dataset.shape[0] - min_train_size - step_size) // step_size

    # Define the first train size
    mapes = []
    rmses = []
    mpes = []
    current_train_size = min_train_size - step_size

    print(f"The number of steps is: {number_of_steps}")
    for step in range(number_of_steps):

        # First create the base training dataset
        train = dataset.iloc[: current_train_size + step_size]
        print(f"Training size: {train.shape[0]}")

        # Create the test set
        test = dataset.iloc[
            current_train_size
            + step_size : current_train_size
            + step_size
            + forecast_horizon
        ]

        predictions = test.copy()
        print(f"test size: {test.shape[0]}")

        # update the train_size
        current_train_size += step_size

        # Now we call the model fuctions, in this case, the base line
        # instantiate the model and set parameters
        model = Prophet(
            interval_width=0.95,
            growth="linear",
            daily_seasonality="auto",
            weekly_seasonality="auto",
            yearly_seasonality="auto",
            seasonality_mode="additive",
        )

        # Fit on historical data
        model.fit(train)

        # Perform the test
        predictions, metrics = test_model(model, test, "y", "yhat", "prophet")

        if plot:
            plot_train_test_result_arima(
                train,
                serie,
                test,
                predictions,
                "SALES",
                "Predicted",
                metrics[5],
                metrics[0],
            )

        mapes.append(metrics[0])
        rmses.append(metrics[1])
        mpes.append(metrics[5])

    return mapes, mpes


# COMMAND ----------


def preprocess_df_for_decomposition(
    predictions_df: pd.DataFrame,
    testing_df: pd.DataFrame,
    serie: str,
    features_list: list,
) -> pd.DataFrame:
    """
    Performs a preprocessing step into the predictions dataframe before calculating the decomposition of the forecast. Changes index, renames and filter features.

    Args:
        predictions_df (pd.DataFrame): Predictions DataFrame output from model.predict.

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready to calculate the decomposition.
    """

    future_df_model_contri = predictions_df.copy().set_index("ds")
    future_df_model_contri["Key"] = serie

    if "y" in future_df_model_contri.columns:
        future_df_model_contri["y"] = testing_df.set_index("ds")["y"]
        features_list_mod = [feature for feature in features_list]
    else:
        features_list_mod = [feature for feature in features_list]
        features_list_mod.remove("y")

    future_df_model_contri = future_df_model_contri.reset_index()

    future_df_model_contri = future_df_model_contri[
        [*features_list_mod, "yhat", "trend", "yearly", "lower_end", "higher_end"]
    ]

    return future_df_model_contri


# COMMAND ----------


def extract_feature_importance(model: prophet.forecaster.Prophet) -> pd.DataFrame:
    """
    Extracts the feature importance for each regressor of the FB Prophet model.

    Args:
        model (prophet.forecaster.Prophet): The FB Prophet model trained on the validation dataset.

    Returns:
        pd.DataFrame: A DataFrame holding the significance for each of the regressor.
    """

    df_coefs = regressor_coefficients(model)[
        ["regressor", "coef_lower", "coef", "coef_upper"]
    ]
    df_coefs["significance"] = df_coefs.apply(
        lambda x: check_feature_significance(x["coef_lower"], x["coef_upper"]), axis=1
    )
    df_coefs = df_coefs[["regressor", "coef", "significance"]]

    return df_coefs


def decompose_prophet_forecast(
    predictions_df: pd.DataFrame,
    testing_df: pd.DataFrame,
    serie: str,
    features_list: list,
) -> pd.DataFrame:
    """
    Decomposes the final validation forecast (last 12 months) of the model into the components: Baseline (trend + seasonality) + individual external regressors contributions.

    Args:
        predictions_df (pd.DataFrame): The DataFrame returned by the model.predict(df), holding all the features.
        serie (str): The name of the serie.
        features_list (list): A List holding the features of the model (for one specific serie)
        decomposition_period (str): The period to decompose ("actuals" or "forecast")

    Returns:
        pd.DataFrame: Prophet model with the external regressor added.
    """

    # preprocess it
    future_df_model_contri = preprocess_df_for_decomposition(
        predictions_df, testing_df, serie, features_list
    )

    future_df_model_contri["Baseline"] = (
        future_df_model_contri["trend"] + future_df_model_contri["yearly"]
    )
    future_df_model_contri["Unexplained_upper"] = (
        future_df_model_contri["higher_end"] - future_df_model_contri["yhat"]
    )
    future_df_model_contri["Unexplained_lower"] = (
        future_df_model_contri["lower_end"] - future_df_model_contri["yhat"]
    )

    return future_df_model_contri


# COMMAND ----------


def get_shap_values_proba(model, train_data, sample_index):
    """
    This function extract the SHAP values (explanatory values) of the model's predictions
    and plots two different forms of visualization: The global Summary plot and the local
    Waterfall plot.

    :param model: The model object that was previously trained and validated.
    :param train_data: The data to train the SHAP explainer and explain.
    :param sample_index: A index number to plot the local Waterfall plot.

    return: The SHAP values array and the Explainer object.
    """
    explainer = shap.TreeExplainer(model=model, data=train_data)

    shap_values = explainer(train_data)

    html = shap.summary_plot(shap_values, train_data, max_display=20)
    display(html)

    shap.initjs()
    shap.plots.waterfall(shap_values[sample_index], max_display=10)

    return shap_values, explainer


# COMMAND ----------


def compare_staging_model(model_to_register: str, new_model_run: str):

    # --------- Get the current staging model ---------
    models_versions = []
    client = mlflow.MlflowClient()
    for mv in client.search_model_versions(filter_string=f"name={model_to_register}"):
        models_versions.append(dict(mv))

    # Get all the models that are in stage (In the majority of the cases it should be only one)
    current_model = [x for x in models_versions if x["current_stage"] == "Staging"][0]
    # Extract the current staging model MAPE
    current_model_mape = mlflow.get_run(current_model["run_id"]).data.metrics[
        model_config["COMPARISON_METRIC"]
    ]
    # Get the new model MAPE
    candidate_model_mape = mlflow.get_run(new_model_run.run_id).data.metrics[
        model_config["COMPARISON_METRIC"]
    ]

    # --------- Compare the models metrics ---------
    if candidate_model_mape < current_model_mape:
        print(
            f"Candidate model has a better {model_config['COMPARISON_METRIC']} than the active model. Switching models..."
        )

        client.transition_model_version_stage(
            name="REGISTER_MODEL_NAME",
            version=new_model_run.version,
            stage="Staging",
        )

        client.transition_model_version_stage(
            name="REGISTER_MODEL_NAME",
            version=current_model["version"],
            stage="Archived",
        )

    else:
        print(
            f"Active model has a better {model_config['COMPARISON_METRIC']} than the candidate model. No changes to be applied."
        )

    print(
        f"Candidate {model_config['COMPARISON_METRIC']} = {candidate_model_mape}\nCurrent = {current_model_mape}"
    )


# COMMAND ----------


def load_staging_model(serie: str, model_type: str, model_name: str, stage_name: str):
    """
    Load the staging model for a given serie and model

    Parameters:
        serie (str): The unique key for each model
        model_type (str):
        model_name (str): the name of the model: "arima" or "prophet"
        stage_name (str): The MLflow Staging name: "Stating" or "Production"

    Returns:
        prophet or pmdarima model: The model ready for .predict()
    """

    # --------- Get the current staging model ---------

    staging_model_uri = f"models:/{model_config['REGISTER_MODEL_NAME']}_{serie}-{model_type}/{stage_name}"
    print(staging_model_uri)

    # Now load the new model to make predictions
    print(
        "Loading registered model version from URI: '{model_uri}'".format(
            model_uri=staging_model_uri
        )
    )
    if model_name == "arima":
        model = mlflow.pmdarima.load_model(staging_model_uri)
    else:
        model = mlflow.prophet.load_model(staging_model_uri)

    return model


# COMMAND ----------


def calculate_residuals(model, train_data: pd.DataFrame) -> list[float]:
    """
    Calculate residuals based on the difference between actual 'y' values and predicted 'yhat' values.

    Args:
        model (YourModelType): The forecasting model.
        train_data (pd.DataFrame): The training dataset.

    Returns:
        List[float]: List of calculated residuals.
    """
    historical_pred = model.predict()
    resid = list(train_data["y"].values - historical_pred["yhat"].values)
    return resid


def fit_auto_regression(resid: list[float], ar_size: int) -> list[float]:
    """
    Fit an AutoRegressive model to the residuals and return rolling coefficients.

    Args:
        resid (List[float]): The list of residuals.
        ar_size (int): Number of lags for AutoRegressive model.

    Returns:
        List[float]: List of rolling coefficients.
    """
    model_resid = AutoReg(resid, lags=ar_size)
    model_resid_fit = model_resid.fit()

    return model_resid_fit


def generate_forecasted_residuals(
    rolling_coefs: list[float],
    resid: list[float],
    window_size: int,
    ar_size: int,
    test_data: pd.DataFrame,
    predictions: pd.DataFrame,
) -> list[float]:
    """
    Generate forecasted residuals using rolling coefficients and past residuals.

    Args:
        rolling_coefs (List[float]): Rolling coefficients from the AutoRegressive model.
        resid (List[float]): List of residuals.
        window_size (int): Size of the forecasting window.
        ar_size (int): Number of lags for AutoRegressive model.
        test_data (pd.DataFrame): The test dataset.
        predictions (pd.DataFrame): Predictions made by the forecasting model.

    Returns:
        List[float]: List of forecasted residuals.
    """
    forecasted_residuals = []
    pred_error = rolling_coefs[0]
    last_n_residuals = resid[len(resid) - window_size :]

    for prediction_index in range(window_size):
        current_residuals_length = len(last_n_residuals)
        lagged_errors = [
            last_n_residuals[i]
            for i in range(
                current_residuals_length - window_size, current_residuals_length
            )
        ]

        for element_index in range(ar_size):
            pred_error += (
                rolling_coefs[element_index + 1]
                * lagged_errors[window_size - element_index - 1]
            )

        forecasted_residuals.append(pred_error)

        future_resid = (test_data["y"].values - predictions["yhat"].values)[
            prediction_index
        ]
        last_n_residuals.append(future_resid)

    return forecasted_residuals


def correct_forecast(forecasted_residuals: list[float], exceptions: list[int]):
    """
    Correct the forecasted residuals by setting values to 0 for specific indices.

    Args:
        forecasted_residuals (List[float]): List of forecasted residuals.
        exceptions (List[int]): List of indices to keep as is.
    """
    for i in range(len(forecasted_residuals)):
        if i not in exceptions:
            forecasted_residuals[i] = 0


def correct_forecast_with_residuals(
    model, train: pd.DataFrame, test: pd.DataFrame, forecast: np.array
) -> tuple:

    window_size = FORECAST_HORIZON
    ar_size = 5
    global resid

    # Assuming model, train, test, and predictions are defined
    resid = calculate_residuals(model, train)

    residual_model = fit_auto_regression(resid, ar_size)
    rolling_coefs = residual_model.params
    print("Residual Coeffs:", rolling_coefs)

    global forecasted_residuals
    forecasted_residuals = generate_forecasted_residuals(
        rolling_coefs, resid, window_size, ar_size, test, forecast
    )

    exceptions = [7, 8, 9]  # Indices to keep as is
    # correct_forecast(forecasted_residuals, exceptions)

    forecast["yhat"] += forecasted_residuals
    real_resid = test["y"].values - forecast["yhat"].values

    return forecast, real_resid


def plot_resid_diagnostics(model, train_data, model_name):
    """
    Plot the residual diagnostics of the fitted model.
    """

    # Calculate residual
    historical_pred = model.predict()
    resid = list(train_data["y"].values - historical_pred["yhat"].values)

    # test normality
    test_result = normaltest(resid)
    if test_result.pvalue < 0.05:
        print(f"The Resid is not Normal!! P-value = %0.2f" % (test_result.pvalue))
    else:
        print(f"The Resid is Normal!! P-value = %0.2f" % (test_result.pvalue))

    fig, axs = plt.subplots(figsize=(8, 6))
    axs.hist(resid)
    axs.set_title(f"{model_name}'s Historical Residual")
    axs.set_xlabel("Residual")
    axs.set_ylabel("Frequency")
    plt.show()

    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(16, 6))
    plot_acf(resid, lags=48, ax=ax1)
    plot_pacf(resid, lags=48, ax=ax2)
    plt.suptitle(f"{model_name}'s Residual Autocorrelation")
    plt.show()


# COMMAND ----------


def make_features_usa(serie, mkt_agg_model_current):
    """
    Performs the Feature Engineering for the United States models.
    Anything related to creating lags features, differencing and feature selection prior to model fitting should be added here.
    """

    # Automatic check if we are dealing with Prophet model or ARIMA model.
    if "SALES" in mkt_agg_model_current.columns:
        mkt_agg_model_current = mkt_agg_model_current[
            [
                "SALES",
                "Key",
                "vehicle_miles_traveled_mill_miles_day",
                "real_gross_domestic_product",
                "light_weight_vehicle_sales",
            ]
        ]

    else:
        mkt_agg_model_current = mkt_agg_model_current[
            [
                "y",
                "ds",
                "Key",
                "vehicle_miles_traveled_mill_miles_day",
                "real_gross_domestic_product",
                "light_weight_vehicle_sales",
            ]
        ]

    # Drop the null rows (because of the shifts)
    mkt_agg_model_current = mkt_agg_model_current.dropna()

    # order the features
    features = list(mkt_agg_model_current.columns)
    sorted_features = sorted(features)
    mkt_agg_model_current = mkt_agg_model_current.reindex(columns=sorted_features)

    return mkt_agg_model_current


# COMMAND ----------

# Dicionary to map features for each series
country_pool_features_mapping_old = {
    "UNITED STATES-Pool": {
        "vehicle_miles_traveled_mill_miles_day": [1, 2, 12],
        "housing_starts_mill": [7, 9, 10],
    },
    "UNITED STATES-Non_Pool": {
        "vehicle_miles_traveled_mill_miles_day": [1],  # , 13],
        "housing_starts_mill": [2, 9],
        "unemployment_percent": [9],
        "real_gdp_bill_chained": [1],
    },
    "CANADA-Pool": {
        "new_car_sales": [1, 2, 7],
        "gdp": [1],  # , 12],
    },
    "UNITED STATES-TOTAL": {
        "vehicle_miles_traveled_mill_miles_day": [1, 2, 12],
        "housing_starts_mill": [7, 9, 10],
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Prophet Functions

# COMMAND ----------


# COMMAND ----------


# COMMAND ----------


def plot_resid_diagnostics(model, train_data, model_name):
    """
    Plot the residual diagnostics of the fitted model.
    """

    # Calculate residual
    historical_pred = model.predict()
    resid = list(train_data["y"].values - historical_pred["yhat"].values)

    fig, axs = plt.subplots(figsize=(8, 6))
    axs.hist(resid)
    axs.set_title(f"{model_name}'s Historical Residual")
    axs.set_xlabel("Residual")
    axs.set_ylabel("Frequency")
    plt.show()

    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(16, 6))
    plot_acf(resid, lags=32, ax=ax1)
    plot_pacf(resid, lags=32, ax=ax2)
    plt.suptitle(f"{model_name}'s Residual Autocorrelation")
    plt.show()


# COMMAND ----------


def make_features_prophet_canada(
    serie: str, mkt_agg_model_current: pd.DataFrame
) -> pd.DataFrame:

    # Add  COVID flag as external variable
    mkt_agg_model_current["covid"] = mkt_agg_model_current["ds"].apply(
        lambda x: 1 if (x.year == 2020) else 0
    )

    if serie == "CANADA-Pool" or serie == "CANADA-Non_Pool":
        # (27/07/2023) for now not using any lags
        # select the model features
        # print(mkt_agg_model_current.columns)
        mkt_agg_model_current["Central_bank_policy_rate_lag_5"] = mkt_agg_model_current[
            "Central_bank_policy_rate"
        ].shift(5)
        mkt_agg_model_current[
            "Interest_rate,_short-term_lag_5"
        ] = mkt_agg_model_current["Interest_rate,_short-term"].shift(5)
        mkt_agg_model_current[
            "Changes_in_inventories,_LCU_lag_1"
        ] = mkt_agg_model_current["Changes_in_inventories,_LCU"].shift(1)
        mkt_agg_model_current[
            "Interest_rate,_short-term_lag_5"
        ] = mkt_agg_model_current["Interest_rate,_short-term"].shift(5)
        mkt_agg_model_current["Money_market_rate_lag_4"] = mkt_agg_model_current[
            "Money_market_rate"
        ].shift(4)
        mkt_agg_model_current[
            "Spread_of_long_over_short_interest_rate_lag_3"
        ] = mkt_agg_model_current["Spread_of_long_over_short_interest_rate"].shift(3)
        mkt_agg_model_current["Unemployment_rate_lag_2"] = mkt_agg_model_current[
            "Unemployment_rate"
        ].shift(2)
        mkt_agg_model_current["Consumption,_total,_LCU_lag_2"] = mkt_agg_model_current[
            "Consumption,_total,_LCU"
        ].shift(2)
        mkt_agg_model_current["Inflation,_CPI,_aop_lag_3"] = mkt_agg_model_current[
            "Inflation,_CPI,_aop"
        ].shift(2)

        mkt_agg_model_current = mkt_agg_model_current[
            [
                "Key",
                "ds",
                "y",
                "Changes_in_inventories,_LCU_lag_1",
                "Central_bank_policy_rate_lag_5",
                "Interest_rate,_short-term_lag_5",
                "Spread_of_long_over_short_interest_rate_lag_3",
                "Consumption,_total,_LCU_lag_2",
                "crude_oil_price",
                "new_car_sales",
            ]
        ]

        mkt_agg_model_current = mkt_agg_model_current.dropna(
            subset=["Central_bank_policy_rate_lag_5"]
        )

    elif serie == "CI-Pipeline":

        mkt_agg_model_current = mkt_agg_model_current[
            ["Key", "y", "feature2", "feature1"]
        ]

    else:
        print("Series not found!!")

    # Drop the null rows (beucase of the shifts)
    mkt_agg_model_current = mkt_agg_model_current.dropna()
    # order the features
    features = list(mkt_agg_model_current.columns)
    sorted_features = sorted(features)
    mkt_agg_model_current = mkt_agg_model_current.reindex(columns=sorted_features)

    return mkt_agg_model_current


# COMMAND ----------


def get_metrics(
    predictions_df: pd.DataFrame,
    true_df: pd.DataFrame,
    true_label: str,
    predicted_label: str,
):
    """
    Performs the metrics calculation from the testing dataframe and the predictions dataframe.

    Args:
        predictions_df (pd.DataFrame): The predictions Dataframe that came from .predict(test)
        true_df (pd.DataFrame): The Testing Dataframe holding the true values
        true_label (str): The name of the true label column
        predicted_label (str): The name of the predictions column

    Returns:
        tuple: A tuple holding all the metrics: MAPE, RMSE, MAE, R2, FB, MPE, WAPE

    Raises:
        error: None
    """

    # Extract the metrics
    model_r2 = round(r2_score(true_df[true_label], predictions_df[predicted_label]), 2)
    model_mape = round(
        mean_absolute_percentage_error(
            true_df[true_label], predictions_df[predicted_label]
        ),
        2,
    )
    model_rmse = round(
        np.sqrt(
            mean_squared_error(true_df[true_label], predictions_df[predicted_label])
        ),
        2,
    )
    model_mae = round(
        mean_absolute_error(true_df[true_label], predictions_df[predicted_label]), 2
    )
    forecast_bias = round(
        ((predictions_df[predicted_label].sum() / true_df[true_label].sum()) - 1), 2
    )
    model_mpe = round(
        mean_percent_error(true_df[true_label], predictions_df[predicted_label])
    )
    model_wape = round(
        simple_wape(true_df[true_label], predictions_df[predicted_label]), 4
    )
    residuals = true_df[true_label].values - predictions_df[predicted_label].values

    metrics_dic = {
        "MAPE": model_mape,
        "RMSE": model_rmse,
        "MAE": model_mae,
        "R2": model_r2,
        "FB": forecast_bias,
        "MPE": model_mpe,
        "WAPE": model_wape,
        "Residuals": residuals,
    }

    return (
        model_mape,
        model_rmse,
        model_mae,
        model_r2,
        forecast_bias,
        model_mpe,
        model_wape,
        residuals,
    )


# COMMAND ----------


def add_regressors_to_prophet(
    model: prophet.forecaster.Prophet, sorted_features: list
) -> prophet.forecaster.Prophet:
    """
    Receives a Prophet Model instance before fitting and add the external regressors to it.

    Args:
        model (prophet.forecaster.Prophet): The Prophet model instance before fitting.
        sorted_features (list): The list of columns on the dataframe. We iterate over the columns to add the regressor.

    Returns:
        prophet.forecaster.Prophet: Prophet model with the external regressor added.
    """

    # add the regressors to the model
    for feature in sorted_features:
        if feature in ["ds", "y", "Key"]:
            pass
        else:
            model.add_regressor(feature)

    return model


# COMMAND ----------


def train_prophet_instance() -> prophet.forecaster.Prophet:
    """
    Trains a Prophet model instance.

    Args:
        None

    Returns:
        None:
    """

    model = Prophet(
        mcmc_samples=500,
        interval_width=0.80,
        growth="linear",
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=True,
    )

    return model


# COMMAND ----------


def run_train_predicted_each_feature_prophet(
    past_dataset: pd.DataFrame, future_dataset_clean: pd.DataFrame
) -> pd.DataFrame:
    """
    This function performs the data manipulation to train the model on the past values + lag values
    in order to predict only the necessary values for the feature.
    For example, our forecast horizon is 12 monhts, and the feature is the lag 10
    so we need to train the model until the 10th month and only predict the last 2 months.

    Steps:
    1 - Concatenate the past past_dataset, which contain the data until the current month, with the future dataset,
    which contains the future values (the lags of the actuals). Then we will have the full table to train the model
    Extract the forecast horizon based on the amount o nans
    2 - Train the model until the last existing value
    3 - Perform the forecast for the amount of months that are missing
    4 - Add those predictions to the future dataset
    5 - Return the future dataset with all future months filled

    Parameters:
        past_dataset (pandas dataframe): The historical dataframe, with traning data
        future_dataset_clean (pandas dataframe): The future-indexed dataframe for the forecast

    Returns:
        future_dataset (with the Features to perform prediction), features (a list of features)
    """

    # -- Step 1
    future_dataset = future_dataset_clean.copy()
    past_dataset = past_dataset.set_index("ds")
    model_dataset = pd.concat([past_dataset, future_dataset], axis=0)
    features = []

    for feature in model_dataset.drop(["y", "Key"], axis=1).columns:
        if feature == "covid":
            pass
        else:
            FEAT_FORECAST_HORIZON = sum(model_dataset[[feature]][feature].isna())
            features.append(feature)

            # Verify if there is a lag 12 - no need to predict
            if FEAT_FORECAST_HORIZON != 0:
                feature_dataset = model_dataset[[feature]].dropna(subset=[feature])

                # -- Step 2
                # Train the ETS on the whole dataset
                model_ets = train_ets(feature_dataset)

                # Perform the test
                # -- Step 3
                predictions_df = pd.DataFrame(
                    model_ets.forecast(steps=FEAT_FORECAST_HORIZON), columns=[feature]
                )

                # -- Step 4
                future_dataset.loc[predictions_df.index, feature] = predictions_df[
                    feature
                ]

            else:
                pass

    future_dataset = future_dataset.reset_index()

    return future_dataset, features


# COMMAND ----------


def make_future_features_prophet(
    serie: str,
    mkt_agg_model_current: pd.DataFrame,
    future_df: pd.DataFrame,
    sorted_features: list,
) -> pd.DataFrame:
    """
    Creates the features for each series
    based on a dictionary that maps the series with the
    respective set of features. Creates the lags for each of the features
    based on the dictionary

    Parameters:
        serie (string): the unique model key
        mkt_agg_model_current (pandas dataframe): the feature dataframe to extract the historical
        feature values
        future_df (pd.DataFrame): The empty dataframe with future dates on the index.
        sorted_features (list): List with the ordered features to reindex them.


    Return:
        future_df_cp (pandas dataframe): The future dataframe to perform the forecast on

    """

    future_df_cp = future_df.copy()

    # Iterate over each of the features
    for variable, lags in country_pool_features_mapping[serie].items():
        print("Feature: ", variable, "Lags:", lags)
        if lags[0] == 0:
            pass
        # Create a list with the existing values for this feature
        for lag in lags:
            print(f"creating lag {lag} for feature {variable}")
            new_column = list(mkt_agg_model_current[variable][-lag:].values)
            # Iterate over the lags for this feature
            for index in range(1, FORECAST_HORIZON_FORECAST - len(new_column) + 1):
                # Append null to this column
                new_column.append(np.nan)

            # Add the column to the future dataframe
            if (len(new_column)) > 24:
                future_df_cp[f"{variable}_lag_{lag}"] = new_column[-24:]
            else:
                future_df_cp[f"{variable}_lag_{lag}"] = new_column

    future_df_cp = future_df_cp.set_index("ds")
    future_df_cp = future_df_cp.reindex(columns=sorted_features)
    return future_df_cp


# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 Arima Functions

# COMMAND ----------


def make_future_features_arima(
    serie: str, mkt_agg_model_current: pd.DataFrame, future_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates the features for each series
    based on a dictionary that maps the series with the
    respective set of features. Creates the lags for each of the features
    based on the dictionary

    Parameters:
        serie (string): the unique model key
        mkt_agg_model_current (pandas dataframe): the feature dataframe to extract the historical
        feature values

    Return:
        fugure_df_cp (pandas dataframe): The future dataframe to perform the forecast on

    """

    future_df_cp = future_df.copy()

    # Iterate over each of the features
    for variable, lags in country_pool_features_mapping[serie].items():
        print("Feature: ", variable, "Lags:", lags)
        if lags[0] == 0:
            pass
        # Create a list with the existing values for this feature
        for lag in lags:
            print(f"creating lag {lag} for feature {variable}")
            new_column = list(mkt_agg_model_current[variable][-lag:].values)
            # Iterate over the lags for this feature
            for index in range(1, FORECAST_HORIZON_FORECAST - len(new_column) + 1):
                # Append null to this column
                new_column.append(np.nan)

            if (len(new_column)) > 24:
                future_df_cp[f"{variable}_lag_{lag}"] = new_column[-24:]
            else:
                future_df_cp[f"{variable}_lag_{lag}"] = new_column

            future_df_cp[f"{variable}_lag_{lag}"] = future_df_cp[
                f"{variable}_lag_{lag}"
            ].diff()

    return future_df_cp


# COMMAND ----------


def make_future_features_arima_rim_size(
    serie: str, mkt_agg_model_current: pd.DataFrame, future_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates the features for each series
    based on a dictionary that maps the series with the
    respective set of features. Creates the lags for each of the features
    based on the dictionary

    Parameters:
        serie (string): the unique model key
        mkt_agg_model_current (pandas dataframe): the feature dataframe to extract the historical
        feature values

    Return:
        fugure_df_cp (pandas dataframe): The future dataframe to perform the forecast on

    """

    future_df_cp = future_df.copy()

    future_df_cp["date"] = future_df_cp.index
    future_df_cp["covid"] = future_df_cp["date"].apply(
        lambda x: 1 if (x.year == 2020) else 0
    )
    future_df_cp = future_df_cp.drop(["date"], axis=1)

    # Iterate over each of the features
    for variable, lags in country_pool_features_mapping_rim_ize["Any"].items():
        # Create a list with the existing values for this feature
        for lag in lags:
            print(f"creating lag {lag} for feature {variable}")
            new_column = list(mkt_agg_model_current[variable][-lag:].values)
            # Iterate over the lags for this feature
            for index in range(1, FORECAST_HORIZON_FORECAST - len(new_column) + 1):
                # Append null to this column
                new_column.append(np.nan)

            # Add the column to the future dataframe
            print(new_column)
            future_df_cp[f"{variable}_lag_{lag}"] = new_column
            # future_df_cp[f'{variable}_lag_{lag}'] = future_df_cp[f'{variable}_lag_{lag}'].diff()
            # future_df_cp = future_df_cp.dropna(subset=[f'{variable}_lag_{lag}'])

    return future_df_cp


# COMMAND ----------


def insert_manual_gdp_forecast(future_df: pd.DataFrame) -> pd.DataFrame():
    """
    Inserts the forecast of GDP into the Future DataFrame, that will be used to perform out-of-sample forecast for United States models.

      future_df (pd.DataFrame): DataFrame with Future months in index.

    Returns:
        pd.DataFrame: future_df with the GDP forecast as a column.
    """

    data_dic = {
        "quarter": [
            "2023Q1",
            "2023Q2",
            "2023Q3",
            "2023Q4",
            "2024Q1",
            "2024Q2",
            "2024Q3",
            "2024Q4",
            "2025Q1",
            "2025Q2",
            "2025Q3",
            "2025Q4",
        ],
        "real_gross_domestic_product": [
            2.0,
            2.4,
            1.3,
            -1,
            -0.8,
            1,
            2.1,
            2.5,
            -0.8,
            1,
            2.1,
            2.5,
        ],  # Conference Board
    }

    gdp_forecast = pd.DataFrame(data=data_dic)
    # Transform from string to Quarter Period Index
    gdp_forecast["quarter"] = pd.PeriodIndex(gdp_forecast.quarter, freq="Q")

    # Create a new column in the future_df to join on Quarter
    future_df["quarter"] = pd.PeriodIndex(future_df.index, freq="Q")
    # Create auxliary dataframe to join
    future_df_aux = future_df.merge(gdp_forecast, on=["quarter"], how="left")
    future_df_aux["date"] = future_df.index
    future_df = future_df_aux.copy()
    # Clean it
    future_df = future_df.set_index("date")
    future_df = future_df.drop(columns=["quarter"])

    return future_df


# COMMAND ----------


def forecast_features_arima(
    past_dataset: pd.DataFrame, future_dataset_clean: pd.DataFrame
) -> pd.DataFrame:
    """
    This function performs the data manipulation to train the model on the past values + lag values
    in order to predict only the necessary values for the feature.
    For example, our forecast horizon is 12 monhts, and the feature is the lag 10
    so we need to train the model until the 10th month and only predict the last 2 months.

    Steps:
    1 - Concatenate the past past_dataset, which contain the data until the current month, with the future dataset,
    which contains the future values (the lags of the actuals). Then we will have the full table to train the model
    Extract the forecast horizon based on the amount o nans
    2 - Train the model until the last existing value
    3 - Perform the forecast for the amount of months that are missing
    4 - Add those predictions to the future dataset
    5 - Return the future dataset with all future months filled

    Parameters:
        past_dataset (pandas dataframe): The historical dataframe, with traning data
        future_dataset_clean (pandas dataframe): The future-indexed dataframe for the forecast

    Returns:
        future_dataset (with the Features to perform prediction), features (a list of features)
    """

    # -- Step 1
    future_dataset = future_dataset_clean.copy()
    model_dataset = pd.concat([past_dataset, future_dataset], axis=0)
    features = []

    for feature in model_dataset.drop([Y_LABEL, "Key"], axis=1).columns:
        if feature == "real_gross_domestic_product":
            pass
        else:
            FEAT_FORECAST_HORIZON = sum(model_dataset[[feature]][feature].isna())
            features.append(feature)

            # Verify if there is a lag 12 - no need to predict
            if FEAT_FORECAST_HORIZON != 0:
                feature_dataset = model_dataset[[feature]].dropna(subset=[feature])

                # -- Step 2
                # Train the ETS on the whole dataset
                model_ets = ets.ExponentialSmoothing(
                    feature_dataset,
                    trend="add",
                    damped_trend=True,
                    seasonal="add",
                    seasonal_periods=12,
                ).fit()

                # Perform the test
                # -- Step 3
                predictions_df = pd.DataFrame(
                    model_ets.forecast(steps=FEAT_FORECAST_HORIZON).values,
                    columns=[feature],
                    index=future_dataset_clean.index,
                )

                # -- Step 4
                future_dataset.loc[predictions_df.index, feature] = predictions_df[
                    feature
                ]

            else:
                pass

    return future_dataset, features


# COMMAND ----------


def reformat_df_to_arima(original_complete_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns and change index to use in the ARIMA model

    Args:
        original_complete_dataset (pd.DataFrame): Original modeling dataframe as saved in the CDL.

    Returns:
        pd.DataFrame: The complete DataFrame formated for ARIMA modeling.

    """
    original_complete_dataset = original_complete_dataset.sort_values(["date", "Key"])
    # original_complete_dataset = original_complete_dataset.rename(columns={"y": "SALES", "ds": "date"})
    original_complete_dataset = original_complete_dataset.set_index("date")

    return original_complete_dataset


# COMMAND ----------


def predict_future_arima(
    model: pmdarima.arima.arima.ARIMA, future_df: pd.DataFrame, forecast_horizon: int
):
    """
    Perform out-of-sample predictions for the
    future forecast horizon, using loaded SARIMAX models

    Parameters:
        model (sarima model): the pmdarima model
        future_df (pandas dataframe): the Pandas DataFrame with the X features to perform the prediction on

    Returns:
        predictions (pandas dataframe): the predictions dataframe
        conf_init (2d array): the confidence interval array
    """

    # Get predictions
    predictions_array, conf_int = model.predict(
        X=future_df.values, n_periods=forecast_horizon, return_conf_int=True, alpha=0.05
    )
    # Transform to dataframe
    predictions = pd.DataFrame(
        predictions_array, index=future_df.index, columns=[PREDICTED_LABEL]
    )

    # join the confidence interval
    predictions = predictions.join(
        pd.DataFrame(
            conf_int, index=predictions.index, columns=["lower_end", "higher_end"]
        ),
        on=predictions.index,
    )

    return predictions


# COMMAND ----------


def predict_future_arima_univariate(
    model: pmdarima.arima.arima.ARIMA, future_df: pd.DataFrame, forecast_horizon: int
):
    """
    Perform out-of-sample predictions for the
    future forecast horizon, using loaded SARIMAX models

    Parameters:
        model (sarima model): the pmdarima model
        future_df (pandas dataframe): the Pandas DataFrame with the X features to perform the prediction on

    Returns:
        predictions (pandas dataframe): the predictions dataframe
        conf_init (2d array): the confidence interval array
    """

    # Get predictions
    predictions_array, conf_int = model.predict(
        n_periods=forecast_horizon, return_conf_int=True, alpha=0.2
    )
    # Transform to dataframe
    predictions = pd.DataFrame(
        predictions_array, index=future_df.index, columns=[PREDICTED_LABEL]
    )

    # join the confidence interval
    predictions = predictions.join(
        pd.DataFrame(
            conf_int, index=predictions.index, columns=["lower_end", "higher_end"]
        ),
        on=predictions.index,
    )
    predictions = predictions.reset_index()
    predictions = predictions.rename(columns={"index": "date"})
    predictions = predictions.set_index("date")

    return predictions


# COMMAND ----------


def visualize_validation_results(
    pred_df: pd.DataFrame, model_mape: float, model_mae: float, stock_name: str
):
    """
    Creates visualizations of the model validation

    Paramters:
        pred_df: DataFrame with true values, predictions and the date column
        model_mape: The validation MAPE
        model_mae: The validation MAE
        model_wape: The validation WAPE

    Returns:
        None
    """

    # logger.info("Vizualizing the results...")

    fig, axs = plt.subplots(figsize=(12, 5))
    # Plot the Actuals
    sns.lineplot(data=pred_df, x="Date", y="Actual", label="Testing values", ax=axs)
    sns.scatterplot(
        data=pred_df,
        x="Date",
        y="Actual",
        ax=axs,
        size="Actual",
        sizes=(80, 80),
        legend=False,
    )

    # Plot the Forecasts
    sns.lineplot(data=pred_df, x="Date", y="Forecast", label="Forecast values", ax=axs)
    sns.scatterplot(
        data=pred_df,
        x="Date",
        y="Forecast",
        ax=axs,
        size="Forecast",
        sizes=(80, 80),
        legend=False,
    )

    axs.set_title(
        f"Default XGBoost {FORECAST_HORIZON} Months Forecast for {stock_name}\nMAPE: {round(model_mape*100, 2)}% | MAE: {model_mae}"
    )
    axs.set_xlabel("Date")
    axs.set_ylabel("R$")

    # plt.savefig(f"./reports/figures/XGBoost_predictions_{dt.datetime.now().date()}_{stock_name}.png")
    # plt.show()
    return fig


# COMMAND ----------


def extract_learning_curves(model, display: bool = False):
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

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle("XGBoost Learning Curves")
    axs[0].plot(learning_results["validation_0"]["rmse"], label="Training")
    axs[0].set_title("RMSE Metric")
    axs[0].set_ylabel("RMSE")
    axs[0].set_xlabel("Iterations")
    axs[0].legend()

    axs[1].plot(learning_results["validation_0"]["logloss"], label="Training")
    axs[1].set_title("Logloss Metric")
    axs[1].set_ylabel("Logloss")
    axs[1].set_xlabel("Iterations")
    axs[1].legend()

    if display:
        plt.show()

    return fig

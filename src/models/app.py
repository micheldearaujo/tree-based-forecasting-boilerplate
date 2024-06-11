import sys
sys.path.insert(0,'.')

from src.utils import *
from csv import writer
from algo_trading import Trader
import plotly.graph_objects as go


def front_end():
    """
    Main function that creates a future dataframe, makes predictions, and prints the predictions.

    Parameters:
        None
    Returns:
        None
    """

    validation_report_df = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, 'validation_stock_prices.csv'))
    predictions_df = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, 'output_stock_prices.csv'), parse_dates=["Date"])
    historical_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])

    st.write("""# Welcome to the Stock Forecaster!
    Here you can have a glance of how the future of your stocks can look like and *simulute* decisions based on that.
    Please keep in mind that this is just a educational tool and you should not perform financial operations based on that.
    """)

    st.sidebar.write("""### Choose your filters here""")

    STOCK_NAME = st.sidebar.selectbox(
        "Which stock do you want to track?",
        ("BOVA11.SA", "BCFF11.SA", "MXRF11.SA", "HGLG11.SA", "ITSA4.SA", "TAEE4.SA",
               "FLRY3.SA", "VALE3.SA", "RAIZ4.SA", "SANB4.SA", "EGIE3.SA", "BBSE3.SA", "CSMG3.SA", "PETR4.SA")
    )

    # get the historical starting date
    hist_start_date = st.sidebar.date_input(
        "From when do you want to the the historical prices?",
        dt.datetime.today().date() - dt.timedelta(days=2*model_config["FORECAST_HORIZON"])
    )

    # Definir as opções de cores para Forecasting e Historical
    forecast_color = st.sidebar.color_picker('Pick a color for the Forecasting', '#1DF7F2')
    historical_color = st.sidebar.color_picker('Pick a color for the Historical', '#ED33FF')
    validation_color = st.sidebar.color_picker('Pick a color for the Testing', '#FD8788')

    ma_1 = st.sidebar.text_input("Amount of day for the minor moving average:", 3)
    try:
        ma_1 = int(ma_1)
    except:
        st.sidebar.error("This value is not an integer. Amount of day for the lower moving average:")

    ma_2 = st.sidebar.text_input("Amount of day for the higher moving average:", 15)
    try:
        ma_2 = int(ma_2)
    except:
        st.sidebar.error("This value is not an integer. Amount of day for the higher moving average:")

    while ma_2 < ma_1:
        st.sidebar.error("The higher moving average must be higher than the lower moving average!")

    st.sidebar.write("""### Nerdzone""")
    st.sidebar.write("Here we have some technical details about the model and the data.")

    st.sidebar.write("#### Model")
    st.sidebar.write("Model name: ", validation_report_df["Model"].values[0])
    st.sidebar.write("Model training date: ", validation_report_df["Training_Date"].values[0])
    st.sidebar.write("Model inference date: ", validation_report_df["Training_Date"].values[0])

    st.sidebar.write("#### Testing Metrics")
    st.sidebar.write("Testing MAPE", validation_report_df["MAPE"].values[0])
    st.sidebar.write("Testing WAPE", validation_report_df["WAPE"].values[0])
    st.sidebar.write("Testing MAE", validation_report_df["MAE"].values[0])
    st.sidebar.write("Testing RMSE", validation_report_df["RMSE"].values[0])

    st.sidebar.write("#### Data")
    st.sidebar.write("Data source: Yahoo Finance")
 
    
    # filter the predictions dataset to only the stock
    predictions_df = predictions_df[predictions_df["Stock"] == STOCK_NAME]
    # filter the validation dataset to only the stock
    validation_report_df = validation_report_df[validation_report_df["Stock"] == STOCK_NAME]

    # filter the historical dataset to only the stock
    historical_df = historical_df[historical_df["Stock"] == STOCK_NAME]
    
    full_df = pd.concat([historical_df, predictions_df], axis=0).reset_index().fillna(0)
    full_df["Class"] = full_df["Close"].apply(lambda x: "Historical" if x > 0 else "Forecast")
    full_df["Price"] = full_df["Close"] + full_df["Forecast"]
    full_df = full_df[['Date', 'Class', 'Price']]

    # Performs algorithmic trading
    trader = Trader(STOCK_NAME, full_df, [], [])
    trader.preprocess_dataset(ma_1, ma_2)
    stock_df_traded = trader.execute_trading(ma_1, ma_2, False)
    #results = trader.evaluate_model(False)


    full_df = pd.concat([full_df, validation_report_df[["Date", "Price", "Class"]]], axis=0)
    full_df["Date"] = pd.to_datetime(full_df["Date"])

    # filter timeframe to display
    full_df = full_df[full_df["Date"] >= pd.to_datetime(hist_start_date)]

    # make the figure using plotly
    fig = px.line(
        full_df,
        x="Date",
        y="Price",
        color="Class",
        symbol="Class",
        title=f"{model_config['FORECAST_HORIZON']-4} days Forecast for {STOCK_NAME}",
        color_discrete_map={'Forecast': forecast_color, 'Historical': historical_color, 'Testing': validation_color}
    )

    

    # # plot it
    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # display the predictions on web
    col1, col2, col3 = st.columns(3)
    col2.metric(label="Mininum price", value=f"R$ {round(predictions_df['Forecast'].min(), 2)}")
    col1.metric(label="Maximum price", value=f"R$ {round(predictions_df['Forecast'].max(), 2)}")
    col3.metric(label="Amplitude", value=f"R$ {round(predictions_df['Forecast'].max() - predictions_df['Forecast'].min(), 2)}",
                delta=f"{100*round((predictions_df['Forecast'].max() - predictions_df['Forecast'].min())/predictions_df['Forecast'].min(), 4)}%")


    st.write("""## Let the algorithm trade for you!""")

    # make the figure using plotly
    fig = px.line(
        full_df[full_df['Class'] != "Testing"],
        x="Date",
        y="Price",
        color="Class",
        symbol="Class",
        title=f"Algorithmic Trading for {STOCK_NAME}",
        color_discrete_map={'Forecast': forecast_color, 'Historical': historical_color, 'Testing': validation_color}
    )

    fig.add_trace(go.Scatter(
        x=full_df["Date"],
        y=full_df[f'SMA_{ma_1}'],
        mode='lines',
        name=f'SMA_{ma_1}'
    ))

    fig.add_trace(go.Scatter(
        x=full_df["Date"],
        y=full_df[f'SMA_{ma_2}'],
        mode='lines',
        name=f'SMA_{ma_2}'
    ))

    fig.add_trace(go.Scatter(
        x=full_df["Date"],
        y=full_df["Buy Signals"],
        mode='markers',
        name='Buy Signals',
        marker=dict(size=15, color='gold',
                    line=dict(color='DarkSlateGrey', width=2))
    ))

    fig.add_trace(go.Scatter(
        x=full_df["Date"],
        y=full_df["Sell Signals"],
        mode='markers',
        name='Sell Signals',
        marker=dict(size=15, color='firebrick',
                    line=dict(color='DarkSlateGrey', width=2))
    ))

    # # plot it
    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Opção para enviar feedback
    feedback = st.text_input("Do you havy any feedback? Let your ideias flow:")
    if feedback:
        st.write("Thanks for your feedback, we will carefully look at your sugestions!")
        
        with open(os.path.join(OUTPUT_DATA_PATH, 'feedbacks.csv'), 'a') as file:
            csv_writer = writer(file)
            csv_writer.writerow([dt.datetime.today(), feedback])

# Execute the whole pipeline
if __name__ == "__main__":

    front_end()
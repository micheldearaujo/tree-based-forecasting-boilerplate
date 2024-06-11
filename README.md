
[![Previsão de ações](https://github.com/micheldearaujo/forecasting_stocks/actions/workflows/main.yml/badge.svg)](https://github.com/micheldearaujo/forecasting_stocks/actions/workflows/main.yml)

# Stock prices forecasting & Algo Trading

The objective of this project is to create a prescriptive model with the aim of informing the user about the best timing to buy or sell stocks based on parameters set by the user.

The product is based on a Forecasting model that will predict the stock price (user's choice) for the next 10 business days. Based on these values it will make recommendations for buying or selling, based on strategies (simple crossing of moving averages).

![alt text](https://github.com/micheldearaujo/forecasting_stocks/blob/main/reports/home_page.png)

## Authors
- [@micheldearaujo](https://github.com/micheldearaujo/forecasting_stocks)
- [Linkedin](https://www.linkedin.com/in/michel-de-ara%C3%BAjo-947377197/)


## Features
This project is composed by the main features:

- Mutable list of ticker symbols. You can add how many ticker symbols you want. Just make sure they follow the Yahoo Finance naming standard.
- Forecasting of each ticker symbol using XGBoost (adding more models is WIP). The features are:
  - Time known features: day, month, year, quarter, week
  - Moving average features. 3 days moving averages default, but you can add how many you want
  - Lag features. Lag 1 default, but you can add how many you want.
- Model testing in the last 10 business days.
- Simple trading algorithm based on crossing of moving averages.
- Front-end page where you can change the timeframe, colors and moving averages of the trading algorithm.

## Installation

Use the following code to install the project locally:

```bash
  git clone https://github.com/micheldearaujo/forecasting_stocks.git
  cd forecasting_stocks
  python3 -m venv forecasting_stocks
  source forecasting_stocks/bin/activate
  make install
```

## Usage/Examples

After installing the project, all one need to do is to run the data pipeline:

```python
# Data pipeline
  python3 src/data/make_dataset.py
  python3 src/features/build_features.py
```

Then, just run the modeling pipeline
```python
  python3 src/models/train_model.py
  python3 src/models/test_model.py
  python3 src/models/predict_model.py
  streamlit run src/models/app.py
```

Last but not least, launch the Streamlit interface:
```bash
  streamlit run src/app/app.py
```

![alt text](https://github.com/micheldearaujo/forecasting_stocks/blob/main/reports/trade.png)
# Stock-price-prediction

BAJFINANCE Stock Price Prediction 📈
Project Overview

This project predicts the Volume Weighted Average Price (VWAP) of BAJFINANCE stock using historical market data. The system leverages time series analysis and machine learning (ARIMA with exogenous features) to forecast future stock prices, providing investors and analysts with insights to make informed trading decisions.

Features

Forecasts BAJFINANCE VWAP using historical stock data

Creates lag features (rolling means and standard deviations) to capture market trends

Handles missing data and preprocesses datasets for modeling

Visualizes actual vs predicted stock prices for intuitive evaluation

Evaluates model performance using metrics: MAE, MSE, RMSE, MAPE, and R²

Technologies Used

Python 3.1.1

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, pmdarima

Model: ARIMA with exogenous features (auto_arima)

Tools: Jupyter Notebook

Installation

Clone the repository:

git clone https://github.com/yourusername/bajfinance-stock-prediction.git


Navigate to the project folder:

cd bajfinance-stock-prediction


Create a virtual environment (optional):

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Note: pmdarima is used for ARIMA modeling:

pip install pmdarima

Data

The dataset contains BAJFINANCE stock data including:
Date, Open, High, Low, Close, VWAP, Volume, Turnover, Trades

Missing values are removed prior to modeling

Lag features (rolling means and rolling standard deviations) are created for:
High, Low, Volume, Turnover, Trades

Rolling windows used: 3-day and 7-day

Usage

Load the dataset:

import pandas as pd
df = pd.read_csv("Datasets/BAJFINANCE.csv")
df.set_index('Date', inplace=True)


Preprocess the data (handle missing values, create lag features).

Split into training and test sets:

training_data = data[0:1800]
test_data = data[1800:]


Train ARIMA model with exogenous features:

from pmdarima import auto_arima
model = auto_arima(y=training_data['VWAP'], X=training_data[ind_features], trace=True)
model.fit(y=training_data['VWAP'], X=training_data[ind_features])


Forecast future VWAP:

forecast = model.predict(n_periods=len(test_data), X=test_data[ind_features])
test_data["Predicted_VWAP"] = forecast


Visualize predictions:

import matplotlib.pyplot as plt
plt.plot(test_data.index, test_data["VWAP"], label="Actual VWAP")
plt.plot(test_data.index, test_data["Predicted_VWAP"], label="Predicted VWAP", linestyle="dashed")
plt.legend()
plt.show()


Evaluate model performance using MAE, MSE, RMSE, MAPE, and R²:

results = evaluate_time_series_model(test_data['VWAP'], forecast)

Model Evaluation

Example results from evaluation:

Metric	Value
MAE	X.XX
MSE	X.XX
RMSE	X.XX
MAPE	X.XX%
R²	X.XX

Replace X.XX with your actual results from evaluate_time_series_model.

Project Structure
bajfinance-stock-prediction/
├── Datasets/             # Stock CSV files
├── notebooks/            # Jupyter notebooks for experiments
├── scripts/              # Scripts for preprocessing and prediction
├── models/               # Trained ARIMA models
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── predict_stock.py      # Main script for predictions

Future Improvements

Add sentiment analysis from financial news or social media to improve accuracy

Experiment with ensemble models (e.g., XGBoost, LSTM) for better predictions

Deploy the model as a web application for interactive stock forecasting

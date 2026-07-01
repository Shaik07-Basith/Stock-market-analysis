# 📊 Stock Market Trend Analysis & Prediction

An interactive Streamlit dashboard for analyzing historical stock data and
forecasting future prices — built as a data analytics portfolio project.

## Features

- **Data Collection** — pulls historical OHLCV data directly from Yahoo Finance via `yfinance`
- **Price Trends** — interactive candlestick chart with 7-day and 30-day moving averages
- **Volatility & Indicators** — daily returns distribution, RSI (Relative Strength Index), MACD
- **Prediction Models**
  - Linear Regression (using lag price + moving averages + RSI as features)
  - ARIMA time-series forecasting
- **Forecasting** — predicts the next N days (configurable, up to 30) of closing prices
- **Raw Data Export** — download the full engineered dataset as CSV

## Project Structure

```
stock_analytics_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup & Installation

1. **Install Python 3.9+** if you don't already have it.

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Your browser will open automatically at `http://localhost:8501`.

## How to Use

1. In the sidebar, enter a stock ticker:
   - US stocks: `AAPL`, `TSLA`, `MSFT`
   - Indian NSE stocks: `TCS.NS`, `RELIANCE.NS`, `INFY.NS`
   - Indices: `^GSPC` (S&P 500), `^NSEI` (Nifty 50)
2. Pick a date range for historical data.
3. Choose a prediction model: **Linear Regression** or **ARIMA**.
4. Set how many days ahead you want to forecast.
5. Explore the four tabs:
   - **Price Trends** — candlestick + moving averages
   - **Volatility & Indicators** — returns distribution, RSI, MACD
   - **Prediction** — model accuracy (RMSE/MAE) + future forecast
   - **Raw Data** — full dataset, downloadable as CSV

## Notes & Extensions

- This is for **educational/analytical purposes only** — not financial advice.
- To extend the project further, you could:
  - Add an LSTM (deep learning) model for sequence prediction
  - Compare multiple stocks/sectors side by side (correlation heatmap)
  - Add news sentiment analysis and correlate it with price movement
  - Deploy the app publicly using Streamlit Community Cloud

## Tech Stack

- Python, Streamlit, Plotly
- pandas, numpy, scikit-learn, statsmodels
- yfinance (Yahoo Finance API wrapper)

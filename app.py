"""
Stock Market Trend Analysis & Prediction App
---------------------------------------------
A Streamlit dashboard that:
  1. Fetches historical stock data (yfinance)
  2. Performs EDA (trends, returns, volatility)
  3. Engineers features (SMA, EMA, RSI, MACD)
  4. Trains prediction models (Linear Regression + ARIMA)
  5. Visualizes results with interactive Plotly charts

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Market Analysis & Prediction", layout="wide")

# ----------------------------------------------------------------------
# SIDEBAR — USER INPUTS
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

ticker = st.sidebar.text_input("Stock Ticker (e.g. TCS.NS, AAPL, RELIANCE.NS)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
model_choice = st.sidebar.selectbox("Prediction Model", ["Linear Regression", "ARIMA"])
forecast_days = st.sidebar.slider("Days to Forecast", 1, 30, 7)

st.title("📊 Stock Market Trend Analysis & Prediction")
st.caption("Analyze historical trends, volatility, and forecast future stock prices.")


# ----------------------------------------------------------------------
# DATA LOADING (cached so repeated interactions don't re-download)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(ticker, start, end):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df.dropna(inplace=True)
    return df


with st.spinner(f"Fetching data for {ticker}..."):
    data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data found. Check the ticker symbol and date range.")
    st.stop()

st.success(f"Loaded {len(data)} rows for {ticker}")


# ----------------------------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------------------------
def add_technical_indicators(df):
    df = df.copy()

    # Moving averages
    df["SMA_7"] = df["Close"].rolling(window=7).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # Daily returns & volatility
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_30"] = df["Daily_Return"].rolling(window=30).std()

    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Lag features (previous day close, useful for regression)
    df["Prev_Close"] = df["Close"].shift(1)

    return df


data = add_technical_indicators(data)

numeric_cols = ["Open","High","Low","Close","Adj Close","Volume"]
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna().reset_index(drop=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Price Trends", "📉 Volatility & Indicators", "🔮 Prediction", "🗂️ Raw Data"]
)

# ----------------------------------------------------------------------
# TAB 1 — PRICE TRENDS (Candlestick + Moving Averages)
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Candlestick Chart with Moving Averages")

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        )
    )
    fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA_7"], name="SMA 7", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA_30"], name="SMA 30", line=dict(width=1)))
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    
    col1.metric("Latest Close", f"{data['Close'].iloc[-1]:.2f}")
    col2.metric(
        "1-Day Change",
        f"{data['Close'].iloc[-1] - data['Close'].iloc[-2]:.2f}",
        f"{data['Daily_Return'].iloc[-1] * 100:.2f}%",
    )
    col3.metric("30-Day Volatility", f"{data['Volatility_30'].iloc[-1]:.4f}")

# ----------------------------------------------------------------------
# TAB 2 — VOLATILITY & TECHNICAL INDICATORS
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Daily Returns Distribution")
    fig_ret = px.histogram(data, x="Daily_Return", nbins=60, marginal="box")
    st.plotly_chart(fig_ret, use_container_width=True)

    st.subheader("RSI (Relative Strength Index)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data["Date"], y=data["RSI_14"], name="RSI 14"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(height=350)
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.subheader("MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data["Date"], y=data["MACD"], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=data["Date"], y=data["MACD_Signal"], name="Signal"))
    fig_macd.update_layout(height=350)
    st.plotly_chart(fig_macd, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 3 — PREDICTION MODELS
# ----------------------------------------------------------------------
with tab3:
    st.subheader(f"Forecast using {model_choice}")

    if model_choice == "Linear Regression":
        model_df = data.dropna(subset=["Prev_Close", "SMA_7", "SMA_30", "RSI_14"]).copy()
        features = ["Prev_Close", "SMA_7", "SMA_30", "RSI_14"]
        X = model_df[features]
        y = model_df["Close"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        col1, col2 = st.columns(2)
        col1.metric("RMSE", f"{rmse:.3f}")
        col2.metric("MAE", f"{mae:.3f}")

        results_df = pd.DataFrame(
            {
                "Date": model_df["Date"].iloc[-len(y_test):].values,
                "Actual": y_test.values,
                "Predicted": y_pred,
            }
        )
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=results_df["Date"], y=results_df["Actual"], name="Actual"))
        fig_pred.add_trace(go.Scatter(x=results_df["Date"], y=results_df["Predicted"], name="Predicted"))
        fig_pred.update_layout(height=450, title="Actual vs Predicted Close Price (Test Set)")
        st.plotly_chart(fig_pred, use_container_width=True)

        # Forecast next N days by iteratively feeding predictions back in
        st.write(f"### Next {forecast_days}-Day Forecast")
        last_row = model_df.iloc[-1].copy()
        future_preds = []
        for _ in range(forecast_days):
            X_next = pd.DataFrame([last_row[features]])
            next_pred = lr_model.predict(X_next)[0]
            future_preds.append(next_pred)
            # roll features forward with the new prediction
            last_row["Prev_Close"] = next_pred
            last_row["SMA_7"] = (last_row["SMA_7"] * 6 + next_pred) / 7
            last_row["SMA_30"] = (last_row["SMA_30"] * 29 + next_pred) / 30

        future_dates = pd.bdate_range(
            start=model_df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days
        )
        forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_preds})
        st.dataframe(forecast_df, use_container_width=True)
        st.line_chart(forecast_df.set_index("Date"))

    else:  # ARIMA
        series = data.set_index("Date")["Close"]
        train_size = int(len(series) * 0.9)
        train, test = series[:train_size], series[train_size:]

        arima_model = ARIMA(train, order=(5, 1, 0))
        arima_fit = arima_model.fit()

        test_pred = arima_fit.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test, test_pred))
        mae = mean_absolute_error(test, test_pred)

        col1, col2 = st.columns(2)
        col1.metric("RMSE", f"{rmse:.3f}")
        col2.metric("MAE", f"{mae:.3f}")

        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual"))
        fig_arima.add_trace(go.Scatter(x=test.index, y=test_pred.values, name="Predicted"))
        fig_arima.update_layout(height=450, title="ARIMA: Actual vs Predicted (Test Set)")
        st.plotly_chart(fig_arima, use_container_width=True)

        # Forecast future days using the full series
        full_model = ARIMA(series, order=(5, 1, 0)).fit()
        future_forecast = full_model.forecast(steps=forecast_days)
        future_dates = pd.bdate_range(
            start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_days
        )
        forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_forecast.values})

        st.write(f"### Next {forecast_days}-Day Forecast")
        st.dataframe(forecast_df, use_container_width=True)
        st.line_chart(forecast_df.set_index("Date"))

# ----------------------------------------------------------------------
# TAB 4 — RAW DATA
# ----------------------------------------------------------------------
with tab4:
    st.subheader("Raw + Engineered Data")
    st.dataframe(data, use_container_width=True)
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, f"{ticker}_data.csv", "text/csv")

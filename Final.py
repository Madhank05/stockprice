import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- PAGE CONFIG ---
st.set_page_config(page_title="🚀 Stock Price Predictor", layout="centered")

# --- CUSTOM CSS FOR STYLING & ANIMATION ---
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
        }
        .main {
            font-family: 'Segoe UI', sans-serif;
            color: #333333;
        }
        .title {
            font-size: 45px;
            font-weight: bold;
            color: #FF4B4B;
            animation: fadein 2s;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 22px;
            font-weight: 500;
            color: #3B3B98;
            text-align: center;
        }
        .section {
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            background-color: #fefefe;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        @keyframes fadein {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        .metric-label {
            font-weight: bold;
            color: #2C3E50;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
df = pd.read_excel('yahoo_data.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.dropna(inplace=True)

# --- LAST YEAR DATA ---
last_date = df['Date'].max()
one_year_ago = last_date - pd.DateOffset(years=1)
one_year_data = df[df['Date'] >= one_year_ago]

# --- TRAIN & TEST SPLIT ---
train_data = one_year_data[:180]
test_data = one_year_data[180:240]

X_train = train_data[['Open', 'High', 'Low', 'Volume']]
y_train = train_data['Close*']
X_test = test_data[['Open', 'High', 'Low', 'Volume']]
y_test = test_data['Close*']

# --- TRAIN MODEL ---
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

test_data = test_data.copy()
test_data['Predicted Close'] = y_pred

# --- HEADER ---
st.markdown('<div class="title">🚀 Stock Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">See predictions, compare with real values, and explore!</div>', unsafe_allow_html=True)

# --- DATE RANGE PICKER ---
st.markdown('<div class="section">', unsafe_allow_html=True)
min_date = test_data['Date'].min().date()
max_date = test_data['Date'].max().date()
st.info(f"📅 Select a range between **{min_date}** and **{max_date}**")

date_range = st.date_input("🗓️ Choose date range", (min_date, max_date), min_value=min_date, max_value=max_date)
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
filtered_data = test_data[(test_data['Date'] >= start_date) & (test_data['Date'] <= end_date)]

if not filtered_data.empty:
    st.subheader("📋 Actual vs Predicted Prices (Selected Range)")
    df_show = filtered_data[['Date', 'Close*', 'Predicted Close']].rename(columns={'Close*': 'Actual Close'})
    df_show = df_show.set_index('Date')
    st.dataframe(df_show.style.format({'Actual Close': '${:.2f}', 'Predicted Close': '${:.2f}'}))
else:
    st.warning("⚠️ No data available for selected date range.")

st.markdown('</div>', unsafe_allow_html=True)

# --- INDIVIDUAL DATE PICKER ---
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("🔍 View Stock Price for a Particular Date")
specific_date = st.date_input("📌 Pick a specific date:", value=min_date, min_value=min_date, max_value=max_date)
specific_row = test_data[test_data['Date'] == pd.to_datetime(specific_date)]

if not specific_row.empty:
    actual = specific_row['Close*'].values[0]
    predicted = specific_row['Predicted Close'].values[0]
    st.success(f"📅 On {specific_date}:")
    st.markdown(f"✅ **Actual Close:** ${actual:.2f}")
    st.markdown(f"🔵 **Predicted Close:** ${predicted:.2f}")
else:
    st.warning("📭 No prediction available for this date.")

st.markdown('</div>', unsafe_allow_html=True)

# --- PLOT CHART ---
with st.expander("📊 View Full Prediction Chart"):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(one_year_data['Date'], one_year_data['Close*'], label='🔴 Actual (Full Year)', linestyle='--')
    ax.plot(train_data['Date'], train_data['Close*'], label='🟢 Training Data')
    ax.plot(test_data['Date'], test_data['Predicted Close'], label='🔵 Predicted (Test)')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_title("📈 Stock Closing Price Prediction")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price ($)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- METRICS ---
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("📉 Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")
with col2:
    st.metric("📊 R-squared", f"{r2_score(y_test, y_pred):.2f}")
st.markdown('</div>', unsafe_allow_html=True)

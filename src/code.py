import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime, timedelta

# 📌 Step 1: Index to ticker mapping
INDEX_MAP = {
    "nifty": "^NSEI",
    "nasdaq": "^IXIC",
    "dowjones": "^DJI",
    "sensex": "^BSESN",
    "sp500": "^GSPC",
}

# 👉 Ask user for input
user_input = input("Enter Index (e.g., nifty, nasdaq, sensex): ").lower()

if user_input not in INDEX_MAP:
    raise ValueError("Unknown index. Try 'nifty', 'nasdaq', 'dowjones', 'sensex', or 'sp500'")

ticker = INDEX_MAP[user_input]
print(f"Using ticker: {ticker} for index '{user_input.upper()}'")

# Parameters
prediction_days = 60
future_days = (datetime(2025, 12, 31) - datetime.today()).days

# 📥 Step 2: Load data
df = yf.download(ticker, start='2015-01-01', end=datetime.today().strftime('%Y-%m-%d'))
df = df[['Close']].dropna()

# 📊 Step 3: Preprocess
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

x_train, y_train = [], []
for i in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[i - prediction_days:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

# 🧠 Step 4: Train model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 🔮 Step 5: Forecast into the future
input_sequence = scaled_data[-prediction_days:]
input_sequence = input_sequence.reshape(1, prediction_days, 1)

future_predictions = []
for _ in range(future_days):
    pred = model.predict(input_sequence)[0, 0]
    future_predictions.append(pred)
    input_sequence = np.append(input_sequence[:, 1:, :], [[[pred]]], axis=1)

# ⏱️ Step 6: Inverse transform & prepare dates
predicted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = [datetime.today() + timedelta(days=i) for i in range(1, future_days + 1)]
predicted_df = pd.DataFrame({'Date': future_dates, 'Predicted': predicted_prices.flatten()})

# 📈 Step 7: Plot
plt.figure(figsize=(14, 6))
plt.plot(df.index[-100:], df['Close'][-100:], label='Actual')
plt.plot(predicted_df['Date'], predicted_df['Predicted'], label='Predicted')
plt.axvline(x=datetime(2025, 12, 31), color='red', linestyle='--', label='Year-End')
plt.title(f'{user_input.upper()} Price Forecast to End of 2025')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 🎯 Step 8: Print Year-End Prediction
year_end_price = predicted_df[predicted_df['Date'] == datetime(2025, 12, 31)]['Predicted']
if not year_end_price.empty:
    print(f"\n🎯 Predicted {user_input.upper()} index price on Dec 31, 2025: {year_end_price.values[0]:.2f}")
else:
    print("Couldn't find prediction exactly on Dec 31 — showing closest date instead.")
    print(predicted_df.tail(1))

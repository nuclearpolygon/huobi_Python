import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import yfinance as yf

from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM, Input

# Configuration
tech_list = ['BTC-USD', 'XRP-USD']
company_name = ["Bitcoin", "Ripple"]
predict_ahead = 25  # Number of candles to predict into the future
window_size = 60    # Number of past candles used for prediction

# Download or load data
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

data_frames = []
for symbol, name in zip(tech_list, company_name):
    f_name = Path(f'{name}.pkl')
    try:
        df = yf.download(symbol, period='max', interval='1m')
        if len(df) == 0:
            print(f'Downloaded data is empty for {symbol}')
            raise Exception
        df.to_pickle(f_name.__str__())
        print(f'Data updated for {symbol}')
    except Exception as e:
        print(f'reading data for {symbol}')
        df = pd.read_pickle(f_name.__str__())
    data_frames.append(df)

# Use only BTC-USD for now
data = data_frames[0][['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Split into training and testing
train_data = scaled_data[:-predict_ahead]

x_train = []
y_train = []

for i in range(window_size, len(train_data) - predict_ahead + 1):
    x_train.append(train_data[i - window_size:i])
    y_train.append(train_data[i:i + predict_ahead, 3])  # index 3 = 'Close'


x_train = np.array(x_train)
y_train = np.array(y_train)

# Build model
model = Sequential([
    Input(shape=(window_size, 5)),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(predict_ahead)
])

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Prepare test data
x_test = []
y_test = scaled_data[-(predict_ahead + window_size):]
y_actual = data['Close'].values[-predict_ahead:]

x_input = y_test[:window_size]
x_test.append(x_input)

x_test = np.array(x_test)

# Predict
predicted_scaled = model.predict(x_test)
predicted = scaler.inverse_transform(
    np.hstack([np.zeros((predict_ahead, 3)),
               predicted_scaled[0].reshape(-1, 1),
               np.zeros((predict_ahead, 1))]))[:, 3]

# Plot
matplotlib.use('TkAgg')
plt.figure(figsize=(12, 6))
plt.plot(range(len(data)), data['Close'], label='Historical Close')
plt.plot(range(len(data) - predict_ahead, len(data)), y_actual, label='Actual Future Close')
plt.plot(range(len(data) - predict_ahead, len(data)), predicted, label='Predicted Future Close')
plt.title('Crypto Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

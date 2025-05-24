import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM, Input, Bidirectional, Dropout, LayerNormalization
from keras.api.callbacks import ModelCheckpoint

# Configuration
pd.options.mode.chained_assignment = None
tech_list = ['BTC-USD', 'XRP-USD']
company_name = ["Bitcoin", "Ripple"]
PREDICT_AHEAD = 25  # Number of candles to predict into the future
WINDOW_SIZE = 60    # Number of past candles used for prediction
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

# Read env
SAVE_CHECKPOINTS = bool(os.getenv('SAVE_CHECKPOINTS'))
TRAIN = bool(os.getenv('TRAIN'))
READ_CHECKPOINTS = bool(os.getenv('READ_CHECKPOINTS'))
READ_PATH = os.getenv('READ_PATH')

def get_checkpoint_path(_next=False, stem='checkpoint') -> Path:
    num = 0
    path = model_dir / f'{stem}_{num}.keras'
    if not path.exists():
        return path
    while path.exists():
        num += 1
        path = model_dir / f'{stem}_{num}.keras'
    if _next:
        return path
    return model_dir / f'{stem}_{num-1}.keras'

# Build model
model = Sequential([
    Input(shape=(WINDOW_SIZE, 9)),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    LayerNormalization(),
    LSTM(64),
    Dropout(0.2),
    Dense(PREDICT_AHEAD)
])
if (get_checkpoint_path().exists() and READ_CHECKPOINTS)\
        or READ_PATH:
    model.load_weights(READ_PATH or get_checkpoint_path())

model.compile(optimizer='adam', loss='mse')

# Download or load data
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

data_frames = []
scaled_data = None
data = None
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
    checkpoint = ModelCheckpoint(filepath=get_checkpoint_path(_next=True),
                                 save_weights_only=False,
                                 save_best_only=True,
                                 monitor='loss',
                                 verbose=1)
    # Use only BTC-USD for now
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    data['RSI'] = RSIIndicator(close=pd.Series(data['Close'].to_numpy().reshape(-1))).rsi().to_numpy()
    data['MACD'] = MACD(close=pd.Series(data['Close'].to_numpy().reshape(-1))).macd().to_numpy()
    data['MA'] = SMAIndicator(close=pd.Series(data['Close'].to_numpy().reshape(-1)), window=14).sma_indicator().to_numpy()
    data.dropna(inplace=True)
    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    data_frames.append((data, scaled_data, scaler))
    # print(scaled_data)
    # Split into training and testing
    train_data = scaled_data[:-PREDICT_AHEAD]
    train_input = []
    train_output = []
    for i in range(WINDOW_SIZE, len(train_data) - PREDICT_AHEAD + 1):
        train_input.append(train_data[i - WINDOW_SIZE:i])
        train_output.append(train_data[i:i + PREDICT_AHEAD, 3])
    train_input = np.array(train_input)
    train_output = np.array(train_output)
    callbacks = []
    if SAVE_CHECKPOINTS:
        callbacks = [checkpoint]
    if not TRAIN:
        continue
    history = model.fit(train_input, train_output, epochs=5, batch_size=32, callbacks=callbacks)
    pd.DataFrame(history.history).to_csv('training_history.csv', mode='a')
# Prepare test data
x_test = []
data, scaled_data, scaler = data_frames[0]
y_test = scaled_data[-(PREDICT_AHEAD + WINDOW_SIZE):]
y_actual = data['Close'].values[-PREDICT_AHEAD:]

x_input = y_test[:WINDOW_SIZE]
# print(df)
# print(df.keys())
# print(df.index)
x_test.append(x_input)

x_test = np.array(x_test)
# Predict
predicted_scaled = model.predict(x_test)
predicted = scaler.inverse_transform(
    np.hstack([np.zeros((PREDICT_AHEAD, 3)),
               predicted_scaled[0].reshape(-1, 1),
               np.zeros((PREDICT_AHEAD, scaled_data.shape[1]-4))]))[:, 3]

# Plot
matplotlib.use('TkAgg')
figure = plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Historical Close')
plt.plot(data.index, data['MA'], label='MA')
# plt.plot(range(len(data)), data['MA'], label='MA')
plt.plot(data.index[-PREDICT_AHEAD:], y_actual, label='Actual Future Close')
plt.plot(data.index[-PREDICT_AHEAD:], predicted, label='Predicted Future Close')
plt.title('Crypto Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['LogReturn'], label='LogReturn')
plt.title('Log Return')
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['MACD'], label='MACD')
plt.title('MACD')
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['RSI'], label='RSI')
plt.title('RSI')
plt.show()
# plt.savefig(get_checkpoint_path(stem='plot').with_suffix('.png'))

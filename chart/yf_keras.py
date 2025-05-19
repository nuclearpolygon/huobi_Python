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
from keras.api.layers import Dense, LSTM, Input, Bidirectional, Dropout, LayerNormalization
from keras.api.callbacks import ModelCheckpoint

# Configuration
tech_list = ['BTC-USD', 'XRP-USD']
company_name = ["Bitcoin", "Ripple"]
PREDICT_AHEAD = 25  # Number of candles to predict into the future
WINDOW_SIZE = 60    # Number of past candles used for prediction
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

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
    Input(shape=(WINDOW_SIZE, 5)),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    LayerNormalization(),
    LSTM(64),
    Dropout(0.2),
    Dense(PREDICT_AHEAD)
])
if get_checkpoint_path().exists():
    model.load_weights(get_checkpoint_path())

model.compile(optimizer='adam', loss='mse')
scaler = MinMaxScaler()

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
    data_frames.append(df)
    checkpoint = ModelCheckpoint(filepath=get_checkpoint_path(_next=True),
                                 save_weights_only=False,
                                 save_best_only=True,
                                 monitor='loss',
                                 verbose=1)
    # Use only BTC-USD for now
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    # Normalize
    scaled_data = scaler.fit_transform(data)
    # Split into training and testing
    train_data = scaled_data[:-PREDICT_AHEAD]
    x_train = []
    y_train = []
    for i in range(WINDOW_SIZE, len(train_data) - PREDICT_AHEAD + 1):
        x_train.append(train_data[i - WINDOW_SIZE:i])
        y_train.append(train_data[i:i + PREDICT_AHEAD, 3])  # index 3 = 'Close'
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    history = model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=[checkpoint])
    pd.DataFrame(history.history).to_csv('training_history.csv', mode='a')
# Prepare test data
x_test = []
y_test = scaled_data[-(PREDICT_AHEAD + WINDOW_SIZE):]
y_actual = data['Close'].values[-PREDICT_AHEAD:]

x_input = y_test[:WINDOW_SIZE]
x_test.append(x_input)

x_test = np.array(x_test)

# Predict
predicted_scaled = model.predict(x_test)
predicted = scaler.inverse_transform(
    np.hstack([np.zeros((PREDICT_AHEAD, 3)),
               predicted_scaled[0].reshape(-1, 1),
               np.zeros((PREDICT_AHEAD, 1))]))[:, 3]

# Plot
matplotlib.use('TkAgg')
plt.figure(figsize=(12, 6))
plt.plot(range(len(data)), data['Close'], label='Historical Close')
plt.plot(range(len(data) - PREDICT_AHEAD, len(data)), y_actual, label='Actual Future Close')
plt.plot(range(len(data) - PREDICT_AHEAD, len(data)), predicted, label='Predicted Future Close')
plt.title('Crypto Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig(get_checkpoint_path(stem='plot').with_suffix('.png'))

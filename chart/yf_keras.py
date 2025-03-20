import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import yfinance as yf

from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

tech_list = ['BTC-USD', 'XRP-USD']

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
BTCUSD: pd.DataFrame
XRPUSD: pd.DataFrame
company_name = ["Bitcoin", "Ripple"]

for i, stock in enumerate(tech_list):
    stock_var = stock.replace('-', '')
    f_name = Path(f'{company_name[i]}.pkl')
    if not f_name.exists():
        globals()[stock_var] = yf.download(stock, period='max', interval='1m')
    else:
        globals()[stock_var] = pd.read_pickle(f_name.__str__())

company_list = [BTCUSD, XRPUSD]
for i, company in enumerate(company_list):
    f_name = Path(f'{company_name[i]}.pkl')
    if not f_name.exists():
        company.to_pickle(f_name.__str__())


ma_day = [10, 20, 50]
for ma in ma_day:
    for i, company in enumerate(company_list):
        column_name = f"MA{ma}"
        company[column_name] = company['Close'].rolling(ma).mean()
        company['Return'] = company['Close'][tech_list[i]].pct_change()


df = pd.concat(company_list, axis=0)

plt.figure(figsize=(15, 10))
for i, symbol in enumerate(company_list, 1):
    ax1 = plt.subplot(2, 1, i)
    plt.title(company_name[i - 1])
    ax1.plot(symbol.index, symbol['Close'], linewidth=2, label='Price')
    ax2 = ax1.twinx()
    ax2.plot(symbol.index, symbol['Volume'], linewidth=2, label='Volume', color='red')
    ax2.grid(False)
    ax2.set_ylim(0, float(symbol['Volume'].max()) * 3)
    ax3 = ax1.twinx()
    ax3.plot(symbol.index, symbol['Return'], linewidth=1, label='Return', linestyle='--')
    ax3.grid(False)
    for ma in ma_day:
        ax1.plot(symbol.index, symbol[f'MA{ma}'], linewidth=2, label=f'MA {ma}')

plt.tight_layout()

sns.jointplot(x='BTC-USD', y='XRP-USD', data=df['Close'])
# plt.show()

data = BTCUSD.filter(['Close'])
dataset = data.values
training_data_len = int(np.ceil( len(dataset) * .95 ))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002
test_data = scaled_data[training_data_len - 60:, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

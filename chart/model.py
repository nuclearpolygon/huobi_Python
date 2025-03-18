import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from huobi.client.market import MarketClient, Candlestick
from huobi.constant import *

from datetime import datetime, timezone, timedelta

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Prepare the Dataset
class CryptoPriceDataset(Dataset):
    def __init__(self, prices, _seq_length, _pred_length):
        self.prices = prices
        self.seq_length = _seq_length
        self.pred_length = _pred_length

    def __len__(self):
        return len(self.prices) - self.seq_length - self.pred_length

    def __getitem__(self, idx):
        # end = (idx + self.seq_length) % len(self.prices)
        # idx = idx % len(self.prices)
        input_seq = self.prices[idx:idx + self.seq_length]
        output_seq = self.prices[idx + self.seq_length:idx + self.seq_length + self.pred_length, 3]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)

# Generate synthetic data (replace this with real cryptocurrency data)
def min_max_normalize(data):
    """
    Normalize a 2D array using Min-Max normalization.
    :param data: 2D array of shape (num_samples, num_features).
    :return: Normalized data, min values, max values.
    """
    data_min = np.min(data, axis=0)[3]
    data_max = np.max(data, axis=0)[3]
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data, data_min, data_max

def min_max_denormalize(normalized_data, data_min, data_max):
    """
    Denormalize a 2D array using Min-Max normalization.
    :param normalized_data: Normalized 2D array.
    :param data_min: Min values used for normalization.
    :param data_max: Max values used for normalization.
    :return: Denormalized data.
    """
    return normalized_data * (data_max - data_min) + data_min

market_client = MarketClient()
def generate_synthetic_data():
    candles = market_client.get_candlestick('btcusdt', CandlestickInterval.MIN1, 2000)
    _data = []
    _x_axis = []
    _plot_data = []
    c: Candlestick
    candles = reversed(candles)
    for c in candles:
        _data.append((c.open, c.high, c.low, c.close, c.vol))
        _plot_data.append(c.close)
        _x_axis.append(datetime.fromtimestamp(c.id, timezone.utc) + timedelta(hours=8))
    _data = np.array(_data, dtype=np.float32)
    _plot_data = np.array(_plot_data)
    return _data, _x_axis, _plot_data

# Parameters
seq_length = 200  # Length of input sequence
pred_length = 30  # Length of predicted sequence
data, x_axis, plot_data = generate_synthetic_data()
_norm_data, _min, _max = min_max_normalize(data)
dataset = CryptoPriceDataset(_norm_data, seq_length, pred_length)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 2: Build the LSTM Model
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, output_size=1, num_layers=10, _pred_length=20):
        super(CryptoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.pred_length = _pred_length
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -self.pred_length:, :])  # Predict the next `pred_length` steps
        out[:, 0, :] = x[:, -1, 3:4]
        return out

model = CryptoLSTM(_pred_length=pred_length).to(device)

# Step 3: Train the Model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)  # Add feature dimension
        targets = targets.unsqueeze(-1).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Step 4: Evaluate the Model
model.eval()
test_input = torch.tensor(_norm_data[-seq_length:-pred_length], dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    predicted_prices = model(test_input).squeeze().cpu().numpy()
predicted_prices = min_max_denormalize(predicted_prices, _min, _max)
# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(x_axis, plot_data, label="Historical Prices")
plt.plot(x_axis[-pred_length:], predicted_prices, label="Predicted Prices", linestyle="--")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y %H:%M'))
plt.gcf().autofmt_xdate()
# plt.axvline(x=x_axis[-pred_length], color="r", linestyle="--", label="Prediction Start")
plt.legend()
plt.show()
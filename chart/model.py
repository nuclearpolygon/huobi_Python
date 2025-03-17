import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from huobi.client.market import MarketClient
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
        output_seq = self.prices[idx + self.seq_length:idx + self.seq_length + self.pred_length]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)

# Generate synthetic data (replace this with real cryptocurrency data)
market_client = MarketClient()
def generate_synthetic_data(length=1000, noise=0.1):
    candles = market_client.get_candlestick('btcusdt', CandlestickInterval.MIN1, 2000)
    _data = []
    _x_axis = []
    for c in candles:
        _data.append(c.close)
        _x_axis.append(datetime.fromtimestamp(c.id, timezone.utc) + timedelta(hours=8))
    _data.reverse()
    _data = np.array(_data)
    _x_axis.reverse()
    prices = (_data - _data.min()) / (_data.max() - _data.min())  # Normalize to [0, 1]
    return prices, _x_axis

# Parameters
seq_length = 50  # Length of input sequence
pred_length = 5  # Length of predicted sequence
data, x_axis = generate_synthetic_data()
dataset = CryptoPriceDataset(data, seq_length, pred_length)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 2: Build the LSTM Model
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=2, _pred_length=20):
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
        inputs = inputs.unsqueeze(-1).to(device)  # Add feature dimension
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
test_input = torch.tensor(data[-seq_length:-pred_length], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
with torch.no_grad():
    predicted_prices = model(test_input).squeeze().cpu().numpy()

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(x_axis, data, label="Historical Prices")
plt.plot(x_axis[-pred_length:], predicted_prices, label="Predicted Prices", linestyle="--")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y %H:%M'))
plt.gcf().autofmt_xdate()
# plt.axvline(x=x_axis[-pred_length], color="r", linestyle="--", label="Prediction Start")
plt.legend()
plt.show()
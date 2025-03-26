import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from huobi.client.market import MarketClient, Candlestick
from huobi.constant import *

from datetime import datetime, timezone, timedelta
from pathlib import Path

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_save_path = "crypto_lstm_model.pth"
market_client = MarketClient()
Intervals = (
    CandlestickInterval.MIN1,
    CandlestickInterval.MIN5,
    CandlestickInterval.MIN15,
    CandlestickInterval.MIN30,
    CandlestickInterval.MIN60,
    CandlestickInterval.HOUR4,
    CandlestickInterval.DAY1,
    CandlestickInterval.WEEK1,
    CandlestickInterval.MON1
)

# Step 1: Prepare the Dataset
class CryptoPriceDataset(Dataset):
    def __init__(self, prices, _seq_length, _pred_length):
        self.prices = prices
        self.seq_length = _seq_length
        self.pred_length = _pred_length

    def __len__(self):
        return len(self.prices) - self.seq_length - self.pred_length

    def __getitem__(self, idx):
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


def generate_synthetic_data(symbol='btcusdt', interval=CandlestickInterval.MIN1):
    candles = market_client.get_candlestick(symbol, interval, 2000)
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


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)  # Linear layer to compute attention scores

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_length, hidden_size)
        attention_scores = self.attention(lstm_output)  # Compute attention scores
        attention_weights = torch.softmax(attention_scores, dim=1)  # Apply softmax to get weights
        context_vector = torch.sum(lstm_output * attention_weights, dim=1)  # Weighted sum
        return context_vector, attention_weights


# Step 2: Build the LSTM Model
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=100, output_size=1, num_layers=10, _pred_length=20, dropout_rate=0.1):
        super(CryptoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.pred_length = _pred_length
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = Attention(hidden_size)  # Add attention layer
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.layer_norm(out)  # Apply layer normalization
        context_vector, attention_weights = self.attention(out)
        out = self.fc(context_vector.unsqueeze(1).repeat(1, self.pred_length, 1))  # Repeat context vector for pred_length steps
        out[:, 0, :] = x[:, -1, 3:4]
        return out, attention_weights


# Parameters
seq_length = 500  # Length of input sequence
pred_length = 30  # Length of predicted sequence
data, x_axis, plot_data = generate_synthetic_data(interval=CandlestickInterval.MIN60)
_norm_data, _min, _max = min_max_normalize(data)
_norm_data = _norm_data[:-pred_length]
dataset = CryptoPriceDataset(_norm_data, seq_length, pred_length)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


model = CryptoLSTM(_pred_length=pred_length).to(device)
if Path(model_save_path).exists():
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
# Step 3: Train the Model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)  # Add feature dimension
        targets = targets.unsqueeze(-1).to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
# Save the model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
# Step 4: Evaluate the Model
model.eval()
test_input = torch.tensor(_norm_data[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    predicted_prices, attention_weights = model(test_input)
    predicted_prices = predicted_prices.squeeze().cpu().numpy()
    attention_weights = attention_weights.squeeze().cpu().numpy()
# Ensure shapes are correct
print("Predicted Prices Shape:", predicted_prices.shape)  # Should be (pred_length,)
print("Attention Weights Shape:", attention_weights.shape)  # Should be (seq_length, pred_length)

# Denormalize the predicted prices
predicted_prices = min_max_denormalize(predicted_prices, _min, _max)
# Visualize the results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(x_axis, plot_data, label="Historical Prices")
plt.plot(x_axis[-pred_length:], predicted_prices, label="Predicted Prices", linestyle="--")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y %H:%M'))
plt.gcf().autofmt_xdate()
plt.axvline(x=x_axis[-pred_length], color="r", linestyle="--", label="Prediction Start")
plt.axvline(x=x_axis[-pred_length-seq_length], color="r", linestyle="--", label="Prediction Start")
plt.legend()

# Plot attention weights
plt.subplot(2, 1, 2)
plt.plot(x_axis[-seq_length-pred_length:-pred_length], attention_weights, label="Predicted Prices", linestyle="--")
# # map_data = np.asarray()
# # sns.heatmap(attention_weights.T.reshape(1, seq_length), cmap="viridis", annot=True, fmt=".2f", cbar=True)
plt.xlabel("Time Steps")
plt.ylabel("Attention Weights")
plt.title("Attention Weights Visualization")

plt.tight_layout()
plt.show()
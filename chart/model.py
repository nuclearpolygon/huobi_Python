import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Step 1: Prepare the Dataset
class CryptoPriceDataset(Dataset):
    def __init__(self, prices, seq_length, pred_length):
        self.prices = prices
        self.seq_length = seq_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.prices) - self.seq_length - self.pred_length

    def __getitem__(self, idx):
        input_seq = self.prices[idx:idx + self.seq_length]
        output_seq = self.prices[idx + self.seq_length:idx + self.seq_length + self.pred_length]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)

# Generate synthetic data (replace this with real cryptocurrency data)
def generate_synthetic_data(length=1000, noise=0.1):
    time = np.arange(0, length)
    prices = np.sin(0.1 * time) + noise * np.random.randn(length)
    prices = (prices - prices.min()) / (prices.max() - prices.min())  # Normalize to [0, 1]
    return prices

# Parameters
seq_length = 50  # Length of input sequence
pred_length = 20  # Length of predicted sequence
data = generate_synthetic_data()
dataset = CryptoPriceDataset(data, seq_length, pred_length)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 2: Build the LSTM Model
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=2):
        super(CryptoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -self.pred_length:, :])  # Predict the next `pred_length` steps
        return out

model = CryptoLSTM()

# Step 3: Train the Model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.unsqueeze(-1)  # Add feature dimension
        targets = targets.unsqueeze(-1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Step 4: Evaluate the Model
model.eval()
test_input = torch.tensor(data[-seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
with torch.no_grad():
    predicted_prices = model(test_input).squeeze().numpy()

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(range(len(data)), data, label="Historical Prices")
plt.plot(range(len(data), len(data) + pred_length), predicted_prices, label="Predicted Prices", linestyle="--")
plt.axvline(x=len(data), color="r", linestyle="--", label="Prediction Start")
plt.legend()
plt.show()
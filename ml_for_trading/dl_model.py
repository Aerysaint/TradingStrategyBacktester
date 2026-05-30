import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, seq_length, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.data = torch.tensor(data.values, dtype=torch.float32).to(device)
        if labels is not None:
            self.labels = torch.tensor(labels.values, dtype=torch.long).to(device)
        else:
            self.labels = None
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        if self.labels is not None:
            y = self.labels[idx + self.seq_length - 1]
            return x, y
        return x

class TradingLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, learning_rate=0.001):
        super(TradingLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 3) # 3 classes: 0 (Hold), 1 (Buy), 2 (Sell)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
    
    def predict_with_uncertainty(self, x, passes=50):
        self.train() # Enable dropout for MC Dropout
        predictions = []
        with torch.no_grad():
            for _ in range(passes):
                preds = self.forward(x)
                predictions.append(preds.unsqueeze(0))
        predictions = torch.cat(predictions, dim=0) # shape: (passes, batch, classes)
        mean_preds = predictions.mean(dim=0)
        variance_preds = predictions.var(dim=0).mean(dim=1) # epistemic uncertainty score
        return mean_preds, variance_preds

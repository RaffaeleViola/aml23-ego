import torch
from torch import nn
from utils.logger import logger
import torch.nn.functional as F  # For the softmax function


class EmgLSTMNet(nn.Module):
    def __init__(self, num_classes=20):
        super(EmgLSTMNet, self).__init__()

        # First LSTM layer: input size 16, output size 5
        self.lstm1 = nn.LSTM(input_size=16, hidden_size=5, batch_first=True)

        # Second LSTM layer: input size 5, output size 50
        self.lstm2 = nn.LSTM(input_size=5, hidden_size=50, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.2)

        # Fully connected (Dense) layer: input size 50, output size 21 classes
        self.fc = nn.Linear(in_features=50, out_features=num_classes)

    def forward(self, x):
        x = x.float()
        # x = x.squeeze(dim=1)
        logger.info(f"Forward step on tensor of size: {x.size()}")
        # First LSTM layer
        lstm1_result, _ = self.lstm1(x)

        # Second LSTM layer
        lstm2_result, _ = self.lstm2(lstm1_result)

        # mid_level_features = {}
        # mid_level_features['features'] = out[:, -1, :]

        # Apply dropout
        droput_result = self.dropout(lstm2_result)

        # Take the last time step output for classification
        last_droput_result = droput_result[:, -1, :]  # shape: (batch_size, hidden_size)

        # Fully connected layer
        result = self.fc(last_droput_result)  # shape: (batch_size, 21)

        return result, last_droput_result  # , mid_level_features

import torch.nn as nn
import torch
from utils.logger import logger
from models import MultiScaleTRN, EmgLSTMNet


class FusionClassifier(nn.Module):
    def __init__(self, input_size=306, hidden_size=256, rgb_frames=5, num_classes=8, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.TRN = MultiScaleTRN(num_frames=rgb_frames)
        self.emgLSTM = EmgLSTMNet()
        self.fusion_layer = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        rgb_data, emg_data = x
        _, rgb_mod = self.TRN(rgb_data)
        _, emg_mod = self.emgLSTM(emg_data)
        emg_mod = self.flatten(emg_mod['features'])
        fused_features = self.fusion_layer(torch.cat((rgb_mod['features'], emg_mod), dim=1))
        out = self.dropout(fused_features)
        out = self.relu(out)
        out = self.fc2(out)
        return out, {}
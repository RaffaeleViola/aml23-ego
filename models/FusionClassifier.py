import torch.nn as nn
import torch


class FusionClassifier(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, num_classes=8, dropout=0.2):
        super().__init__()

        self.fusion_layer = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        rgb_mod, emg_mod = x
        emg_mod = self.flatten(emg_mod)
        fused_features = self.fusion_layer(torch.cat((rgb_mod, emg_mod), dim=1))
        out = self.dropout(fused_features)
        out = self.relu(out)
        out = self.fc2(out)
        return out, {}
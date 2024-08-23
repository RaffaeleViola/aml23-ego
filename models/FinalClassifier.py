import torch
import torch.nn as nn
from models.I3D import InceptionI3d


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

    def forward(self, x):
        return self.classifier(x), {}
    
class I3D_Classifier(nn.Module):
    def __init__(self, num_classes=8, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.logits = InceptionI3d.Unit3D(
            in_channels=1024,
            output_channels=num_classes,
        )

    def forward(self, x):
        x = self.dropout(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        logits = self.logits(x).squeeze(3).squeeze(3).squeeze(2)
        return logits, {}


class MLP_Classifier(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, num_classes=8, dropout=0.5):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.temporal_pooling = nn.AdaptiveMaxPool1d(1)  # Pooling lungo l'asse temporale

    def forward(self, x):

        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        
        out = self.relu(self.fc2(out))
        out = self.dropout(out)

        out = out.permute(0, 2, 1)
        out = self.temporal_pooling(out).squeeze(-1)  
        
        out = self.fc3(out)  
        return out, {}
    
class LSTM_Classifier(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, num_classes=8, dropout=0.5):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)
        self.temporal_pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):        
        out, _ = self.lstm(x)  
        
        out = out[:, -1, :]

        out = self.dropout(out)
        out = self.fc(out)  
        out = self.relu(out)
        
        return out, {}
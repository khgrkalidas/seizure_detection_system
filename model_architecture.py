import torch
import torch.nn as nn


class SeizureModel(nn.Module):

    def __init__(self):
        super().__init__()

        # CNN
        self.conv1 = nn.Conv1d(23, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)

        self.relu = nn.ReLU()

        # Channel attention (MATCHES WEIGHTS)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128, 128, 1),
            nn.Sigmoid()
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        # classifier
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        # CNN
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # attention
        w = self.attn(x)
        x = x * w

        # sequence format
        x = x.permute(0,2,1)

        # LSTM
        x,_ = self.lstm(x)

        # transformer
        x = self.transformer(x)

        # global pooling
        x = x.mean(dim=1)

        # classifier
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x.squeeze(-1)
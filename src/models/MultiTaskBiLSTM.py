import torch
import torch.nn as nn


class MultiTaskBiLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

        self.regressor = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        # ORIGINAL MEAN POOLING
        features = torch.mean(lstm_out, dim=1)

        features = self.dropout(features)

        logits = self.classifier(features)

        rep_out = self.regressor(features)

        return logits, rep_out
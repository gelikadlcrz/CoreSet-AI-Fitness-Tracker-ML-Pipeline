import torch
import torch.nn as nn

class BaselineBiLSTM(nn.Module):
    """
    Baseline Bidirectional LSTM for exercise classification.
    Treats flattened relative joint angles as a sequential temporal series.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=4, dropout=0.5):
        super(BaselineBiLSTM, self).__init__()
        
        # Methodology: Two stacked bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # The hidden_size is doubled because the model is bidirectional
        self.fc_head = nn.Sequential(
            nn.Dropout(dropout), # Methodology: Dropout 0.5 before final layer
            nn.Linear(hidden_size * 2, num_classes)
        )

    def forward(self, x):
        # x shape: [Batch, Sequence_Length, Features]
        
        # LSTM returns (output, (h_n, c_n))
        lstm_out, _ = self.lstm(x)
        
        # We take the output of the last time step for classification
        # Since it's bidirectional, we look at the last frame's representation
        last_time_step = lstm_out[:, -1, :]
        
        # Final classification via the softmax-ready logits
        logits = self.fc_head(last_time_step)
        
        return logits
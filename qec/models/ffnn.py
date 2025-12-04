"""
Feed-Forward Neural Network Model for Quantum Error Correction
"""
import torch
import torch.nn as nn


class ECC_FFNN(nn.Module):
    """
    Feed-Forward Neural Network Decoder.

    Simple baseline model: syndrome → hidden → 4 classes
    Input: flat syndrome vector (dense, no spatial structure)
    """

    def __init__(self, args, dropout=0.0, label_smoothing=0.0):
        super(ECC_FFNN, self).__init__()
        self.args = args

        code = args.code
        input_size = code.pc_matrix.size(0)  # syndrome length
        hidden_size = getattr(args, 'hidden_size', 256)

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4)
        )

        # Loss
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input syndrome tensor (batch_size, syndrome_length)

        Returns:
            logits: (batch_size, 4)
        """
        return self.net(x)

    def loss(self, pred, true_label):
        """Calculate loss."""
        return self.criterion(pred, true_label)


class ECC_FFNN_Large(nn.Module):
    """
    Larger Feed-Forward Neural Network Decoder.

    Deeper network with more hidden units.
    """

    def __init__(self, args, dropout=0.0, label_smoothing=0.0):
        super(ECC_FFNN_Large, self).__init__()
        self.args = args

        code = args.code
        input_size = code.pc_matrix.size(0)
        hidden_size = getattr(args, 'hidden_size', 512)

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 4)
        )

        # Loss
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input syndrome tensor (batch_size, syndrome_length)

        Returns:
            logits: (batch_size, 4)
        """
        return self.net(x)

    def loss(self, pred, true_label):
        """Calculate loss."""
        return self.criterion(pred, true_label)

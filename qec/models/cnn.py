"""
CNN Decoder for Quantum Error Correction

Simple CNN architecture for surface code decoding.
Takes syndrome as 2D grid input and predicts logical error class.
"""
import torch
import torch.nn as nn
import numpy as np


class ECC_CNN(nn.Module):
    """
    CNN-based decoder for surface code.

    Architecture:
    - Input: syndrome reshaped as 2-channel 2D grid (Z and X stabilizers)
    - Multiple conv layers with increasing channels
    - Global average pooling
    - FC layer to 4 classes (I, X, Z, Y)
    """

    def __init__(self, args, dropout=0.0, label_smoothing=0.0):
        super(ECC_CNN, self).__init__()
        self.args = args
        self.L = args.code_L

        # Stabilizer grid sizes
        # For surface code L: approximately (L-1) x (L-1) for each type
        # But we'll use the actual H matrix dimensions
        code = args.code
        self.n_z = code.H_z.shape[0]  # Number of Z stabilizers
        self.n_x = code.H_x.shape[0]  # Number of X stabilizers

        # Compute grid size (assume roughly square)
        # Z and X stabilizers each form a grid
        self.grid_size = self.L  # Use L as grid size

        # CNN layers
        # Input: 2 channels (Z stabilizers, X stabilizers)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Global average pooling + FC
        self.fc = nn.Linear(128, 4)

        # Loss
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _syndrome_to_grid(self, syndrome):
        """
        Convert flat syndrome to 2-channel 2D grid.

        Args:
            syndrome: (batch, n_z + n_x)

        Returns:
            grid: (batch, 2, H, W)
        """
        batch_size = syndrome.shape[0]

        # Split into Z and X syndromes
        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        # Reshape to 2D grids
        # Pad if necessary to make square
        h = w = self.grid_size

        # Pad s_z and s_x to fit grid
        s_z_padded = torch.zeros(batch_size, h * w, device=syndrome.device)
        s_x_padded = torch.zeros(batch_size, h * w, device=syndrome.device)

        s_z_padded[:, :self.n_z] = s_z
        s_x_padded[:, :self.n_x] = s_x

        # Reshape to 2D
        z_grid = s_z_padded.view(batch_size, 1, h, w)
        x_grid = s_x_padded.view(batch_size, 1, h, w)

        # Stack as 2 channels
        grid = torch.cat([z_grid, x_grid], dim=1)

        return grid

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: syndrome tensor (batch, syndrome_len)

        Returns:
            logits: (batch, 4)
        """
        # Convert to 2D grid
        x = self._syndrome_to_grid(x)

        # Conv layers
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        # Global average pooling
        x = x.mean(dim=[2, 3])  # (batch, 128)

        # FC
        x = self.fc(x)

        return x

    def loss(self, pred, true_label):
        """Calculate loss."""
        return self.criterion(pred, true_label)


class ECC_CNN_Large(nn.Module):
    """
    Larger CNN for bigger code distances.
    """

    def __init__(self, args, dropout=0.0, label_smoothing=0.0):
        super(ECC_CNN_Large, self).__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        self.n_z = code.H_z.shape[0]
        self.n_x = code.H_x.shape[0]
        self.grid_size = self.L

        # Deeper CNN
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)
        )

        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _syndrome_to_grid(self, syndrome):
        batch_size = syndrome.shape[0]
        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        h = w = self.grid_size
        s_z_padded = torch.zeros(batch_size, h * w, device=syndrome.device)
        s_x_padded = torch.zeros(batch_size, h * w, device=syndrome.device)
        s_z_padded[:, :self.n_z] = s_z
        s_x_padded[:, :self.n_x] = s_x

        z_grid = s_z_padded.view(batch_size, 1, h, w)
        x_grid = s_x_padded.view(batch_size, 1, h, w)

        return torch.cat([z_grid, x_grid], dim=1)

    def forward(self, x):
        x = self._syndrome_to_grid(x)
        x = self.features(x)
        x = x.mean(dim=[2, 3])
        x = self.classifier(x)
        return x

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)

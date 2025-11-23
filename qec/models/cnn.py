"""
CNN Decoder for Quantum Error Correction

CNN architecture for rotated surface code decoding with accurate stabilizer mapping.
Takes syndrome as 2D grid input and predicts logical error class.
"""
import torch
import torch.nn as nn
import numpy as np


def get_rotated_surface_code_stabilizer_coords(L):
    """
    Rotated surface code의 stabilizer 좌표 계산.

    Rotated surface code L:
    - 큐빗: L x L grid
    - Z stabilizers: 체커보드 패턴의 플라켓 (X error 감지)
    - X stabilizers: 체커보드 패턴의 플라켓 (Z error 감지)

    Stabilizer grid는 (L-1) x (L-1) 내부 + boundary

    Returns:
        z_coords: dict {stabilizer_idx: (row, col)}
        x_coords: dict {stabilizer_idx: (row, col)}
        grid_size: 2D grid 크기
    """
    # 2D grid에 stabilizer 배치
    # Grid size: L x L (큐빗과 같은 크기로 매핑)
    grid_size = L

    z_coords = {}
    x_coords = {}

    # H matrix에서 stabilizer가 어떤 큐빗을 터치하는지 분석해서
    # stabilizer의 "중심" 위치를 계산

    # Rotated surface code에서:
    # - 내부 플라켓: 4개 큐빗 터치
    # - Boundary 플라켓: 2개 큐빗 터치

    # Z stabilizers 배치 (L=3 기준 분석 결과)
    # Z0: [0,1,3,4] -> 중심 (0.5, 0.5) -> grid (0, 0)
    # Z1: [4,5,7,8] -> 중심 (1.5, 1.5) -> grid (1, 1)
    # Z2: [2,5] -> 오른쪽 boundary -> grid (0, 2) 또는 (1, 2)
    # Z3: [3,6] -> 왼쪽 boundary -> grid (1, 0) 또는 (2, 0)

    # 일반화: H matrix에서 직접 위치 계산
    return z_coords, x_coords, grid_size


def compute_stabilizer_positions_from_H(H, L):
    """
    H matrix에서 stabilizer의 2D 위치를 계산.

    각 stabilizer가 터치하는 큐빗들의 최소 좌표를 사용하여 고유 위치 보장.

    Args:
        H: parity check matrix (n_stabilizers, n_qubits)
        L: code distance

    Returns:
        coords: dict {stabilizer_idx: (row, col)}
    """
    coords = {}
    n_stab = H.shape[0]

    for stab_idx in range(n_stab):
        # 이 stabilizer가 터치하는 큐빗들
        qubits = np.where(H[stab_idx] == 1)[0]

        if len(qubits) == 0:
            continue

        # 큐빗들의 2D 좌표
        rows = [q // L for q in qubits]
        cols = [q % L for q in qubits]

        # 최소 좌표 사용 (plaquette의 top-left corner)
        min_row = min(rows)
        min_col = min(cols)

        coords[stab_idx] = (min_row, min_col)

    return coords


class ECC_CNN(nn.Module):
    """
    CNN-based decoder for rotated surface code with accurate stabilizer mapping.

    Architecture:
    - Input: syndrome mapped to 2D grid (L x L)
      - Channel 0: Z stabilizers (detect X errors)
      - Channel 1: X stabilizers (detect Z errors)
    - Multiple conv layers with batch norm
    - Global average pooling
    - FC layer to 4 classes (I, X, Z, Y)
    """

    def __init__(self, args, dropout=0.0, label_smoothing=0.0):
        super(ECC_CNN, self).__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        self.n_z = code.H_z.shape[0]  # Number of Z stabilizers
        self.n_x = code.H_x.shape[0]  # Number of X stabilizers

        # Grid size is L x L for rotated surface code
        self.grid_size = self.L

        # Compute stabilizer coordinates from H matrices
        H_z_np = code.H_z.cpu().numpy() if torch.is_tensor(code.H_z) else code.H_z
        H_x_np = code.H_x.cpu().numpy() if torch.is_tensor(code.H_x) else code.H_x

        self.z_coord_map = compute_stabilizer_positions_from_H(H_z_np, self.L)
        self.x_coord_map = compute_stabilizer_positions_from_H(H_x_np, self.L)

        # CNN layers
        # Input: 2 channels (Z stabilizers, X stabilizers) on L x L grid
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
        Convert flat syndrome to 2-channel 2D grid with accurate stabilizer positions.

        Args:
            syndrome: (batch, n_z + n_x)

        Returns:
            grid: (batch, 2, L, L)
        """
        batch_size = syndrome.shape[0]
        device = syndrome.device
        h = w = self.grid_size

        # Initialize grids
        z_grid = torch.zeros(batch_size, h, w, device=device)
        x_grid = torch.zeros(batch_size, h, w, device=device)

        # Split syndrome into Z and X parts
        s_z = syndrome[:, :self.n_z]  # Z stabilizer syndromes
        s_x = syndrome[:, self.n_z:]  # X stabilizer syndromes

        # Place Z stabilizers at computed positions (accumulate, don't overwrite)
        for idx, (row, col) in self.z_coord_map.items():
            if idx < self.n_z:
                z_grid[:, row, col] += s_z[:, idx]

        # Place X stabilizers at computed positions (accumulate, don't overwrite)
        for idx, (row, col) in self.x_coord_map.items():
            if idx < self.n_x:
                x_grid[:, row, col] += s_x[:, idx]

        # Stack as 2 channels: (batch, 2, H, W)
        grid = torch.stack([z_grid, x_grid], dim=1)

        return grid

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: syndrome tensor (batch, syndrome_len)

        Returns:
            logits: (batch, 4)
        """
        # Convert to 2D grid with accurate positions
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
    Larger CNN for bigger code distances with accurate stabilizer mapping.
    Includes ResNet-style skip connections.
    """

    def __init__(self, args, dropout=0.0, label_smoothing=0.0):
        super(ECC_CNN_Large, self).__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        self.n_z = code.H_z.shape[0]
        self.n_x = code.H_x.shape[0]
        self.grid_size = self.L

        # Compute stabilizer coordinates from H matrices
        H_z_np = code.H_z.cpu().numpy() if torch.is_tensor(code.H_z) else code.H_z
        H_x_np = code.H_x.cpu().numpy() if torch.is_tensor(code.H_x) else code.H_x

        self.z_coord_map = compute_stabilizer_positions_from_H(H_z_np, self.L)
        self.x_coord_map = compute_stabilizer_positions_from_H(H_x_np, self.L)

        # Initial conv to increase channels
        self.conv_in = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(64)

        # ResNet-style blocks
        self.block1 = ResBlock(64, 64, dropout)
        self.block2 = ResBlock(64, 128, dropout)
        self.block3 = ResBlock(128, 256, dropout)

        # Classifier
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
        """Convert flat syndrome to 2-channel 2D grid with accurate positions."""
        batch_size = syndrome.shape[0]
        device = syndrome.device
        h = w = self.grid_size

        z_grid = torch.zeros(batch_size, h, w, device=device)
        x_grid = torch.zeros(batch_size, h, w, device=device)

        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        for idx, (row, col) in self.z_coord_map.items():
            if idx < self.n_z:
                z_grid[:, row, col] += s_z[:, idx]

        for idx, (row, col) in self.x_coord_map.items():
            if idx < self.n_x:
                x_grid[:, row, col] += s_x[:, idx]

        return torch.stack([z_grid, x_grid], dim=1)

    def forward(self, x):
        x = self._syndrome_to_grid(x)

        # Initial conv
        x = torch.relu(self.bn_in(self.conv_in(x)))

        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Global average pooling
        x = x.mean(dim=[2, 3])

        # Classifier
        x = self.classifier(x)

        return x

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)


class ResBlock(nn.Module):
    """Residual block with optional channel change."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Skip connection with 1x1 conv if channels change
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        out = out + identity
        out = torch.relu(out)

        return out

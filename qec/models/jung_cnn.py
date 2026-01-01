import torch
import torch.nn as nn
import numpy as np

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


class JungCNNDecoder(nn.Module):
    """
    CNN-based decoder based on Jung et al., IEEE TQE 2024.
    This model implements the architecture described in the paper
    for high-level decoding (output: 4 classes).
    """
    def __init__(self, args, dropout=0.0, label_smoothing=0.0):
        super().__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        self.n_z = code.H_z.shape[0]  # Number of Z stabilizers
        self.n_x = code.H_x.shape[0]  # Number of X stabilizers

        # The paper uses an (L+1)x(L+1) grid for syndrome input.
        # Stabilizer positions are mapped onto this grid.
        self.grid_size_h = self.L + 1
        self.grid_size_w = self.L + 1

        # Compute stabilizer coordinates from H matrices (same logic as ECC_CNN)
        H_z_np = code.H_z.cpu().numpy() if torch.is_tensor(code.H_z) else code.H_z
        H_x_np = code.H_x.cpu().numpy() if torch.is_tensor(code.H_x) else code.H_x

        self.z_coord_map = compute_stabilizer_positions_from_H(H_z_np, self.L)
        self.x_coord_map = compute_stabilizer_positions_from_H(H_x_np, self.L)
        
        # Hyperparameter settings based on Jung et al. (3) and Table 1, Section V
        if self.L == 3:
            n_filters = 8
        elif self.L == 5:
            n_filters = 32
        else: # L >= 7
            n_filters = 64
            
        # Layer 1: Filter size (3,3) [cite: 423]
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()
        
        # Layer 2: Filter size (2,2) [cite: 423]
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        
        # Calculate Flatten dimension based on output of conv2
        # Input to conv2: (Batch, n_filters, L+1, L+1)
        # Output H/W size after conv2(kernel=2, padding=1, stride=1):
        # H_out = (H_in + 2*padding - kernel_size) / stride + 1
        # H_out = (L+1 + 2*1 - 2) / 1 + 1 = L+1+2-2+1 = L+2
        # So, output shape is (Batch, n_filters, L+2, L+2)
        self.flatten_dim = n_filters * (self.L + 2) * (self.L + 2)
        
        # Dense Layer: 50 nodes (paper fixed value) [cite: 426]
        self.fc1 = nn.Linear(self.flatten_dim, 50)
        
        # Output Layer: 4 nodes (I, X, Y, Z) [cite: 427]
        self.fc2 = nn.Linear(50, 4)

        # Loss function
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _syndrome_to_grid(self, syndrome):
        """
        Convert flat syndrome to 2-channel 2D grid with accurate stabilizer positions.
        
        [Jung et al. 2024 Implementation Detail]
        - Background (No Stabilizer): -0.5 (Incoherent Value)
        - Syndrome 0 (No Error): 0.0
        - Syndrome 1 (Error): 1.0
        """
        batch_size = syndrome.shape[0]
        device = syndrome.device
        h = self.grid_size_h
        w = self.grid_size_w

        # [수정 1] 배경을 0이 아니라 -0.5로 초기화 (논문의 핵심)
        z_grid = torch.full((batch_size, h, w), -0.5, device=device)
        x_grid = torch.full((batch_size, h, w), -0.5, device=device)

        # Split syndrome into Z and X parts
        s_z = syndrome[:, :self.n_z]  
        s_x = syndrome[:, self.n_z:]  

        # [수정 2] 신드롬 값을 -1/1로 변환하지 않고 0/1 그대로 사용
        # (논문: syndrome values are "0" and "1")
        s_z_encoded = s_z.float() 
        s_x_encoded = s_x.float()

        # Place Z stabilizers
        for idx, (row, col) in self.z_coord_map.items():
            if idx < self.n_z:
                if row < h and col < w:
                    z_grid[:, row, col] = s_z_encoded[:, idx]

        # Place X stabilizers
        for idx, (row, col) in self.x_coord_map.items():
            if idx < self.n_x:
                if row < h and col < w:
                    x_grid[:, row, col] = s_x_encoded[:, idx]

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
        
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense Layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x) # Logits 출력
        
        return x

    def loss(self, pred, true_label):
        """Calculate loss."""
        return self.criterion(pred, true_label)
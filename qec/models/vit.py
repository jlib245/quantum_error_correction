"""
Vision Transformer (ViT) style Decoder for Quantum Error Correction

Syndrome → 2D Grid → CNN Patch Embedding → Transformer → Classifier
"""
import torch
import torch.nn as nn
import numpy as np


def compute_stabilizer_positions_from_H(H, L):
    """H matrix에서 stabilizer의 2D 위치를 계산."""
    coords = {}
    n_stab = H.shape[0]

    for stab_idx in range(n_stab):
        qubits = np.where(H[stab_idx] == 1)[0]
        if len(qubits) == 0:
            continue
        rows = [q // L for q in qubits]
        cols = [q % L for q in qubits]
        min_row = min(rows)
        min_col = min(cols)
        coords[stab_idx] = (min_row, min_col)

    return coords


class ECC_ViT(nn.Module):
    """
    Vision Transformer style decoder with CNN patch embedding.

    Flow:
    Syndrome → 2D Grid → Patch Embedding (CNN) → Transformer → Classifier
    """

    def __init__(self, args, dropout=0.1, label_smoothing=0.0):
        super(ECC_ViT, self).__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        self.n_z = code.H_z.shape[0]
        self.n_x = code.H_x.shape[0]
        self.grid_size = self.L

        # Compute stabilizer coordinates
        H_z_np = code.H_z.cpu().numpy() if torch.is_tensor(code.H_z) else code.H_z
        H_x_np = code.H_x.cpu().numpy() if torch.is_tensor(code.H_x) else code.H_x
        self.z_coord_map = compute_stabilizer_positions_from_H(H_z_np, self.L)
        self.x_coord_map = compute_stabilizer_positions_from_H(H_x_np, self.L)

        # Transformer config
        d_model = getattr(args, 'd_model', 128)
        n_heads = getattr(args, 'h', 8)
        n_layers = getattr(args, 'N_dec', 6)

        self.d_model = d_model
        self.n_patches = self.L * self.L  # patch_size=1

        # CNN Patch Embedding (각 위치를 d_model 차원으로)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=1),  # 1x1 conv to project
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.n_patches, d_model) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4)
        )

        # Loss
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _syndrome_to_grid(self, syndrome):
        """
        Convert flat syndrome to 2-channel 2D grid.

        Value encoding: 0 = empty, -1 = no error, +1 = error detected
        """
        batch_size = syndrome.shape[0]
        device = syndrome.device
        h = w = self.grid_size

        z_grid = torch.zeros(batch_size, h, w, device=device)
        x_grid = torch.zeros(batch_size, h, w, device=device)

        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        # Convert syndrome: 0 -> -1, 1 -> +1
        s_z_encoded = s_z * 2 - 1
        s_x_encoded = s_x * 2 - 1

        for idx, (row, col) in self.z_coord_map.items():
            if idx < self.n_z:
                z_grid[:, row, col] = s_z_encoded[:, idx]

        for idx, (row, col) in self.x_coord_map.items():
            if idx < self.n_x:
                x_grid[:, row, col] = s_x_encoded[:, idx]

        return torch.stack([z_grid, x_grid], dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        # Syndrome to 2D grid
        x = self._syndrome_to_grid(x)  # (B, 2, L, L)

        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, L, L)

        # Flatten patches to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, L*L, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+L*L, d_model)

        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Use CLS token for classification
        x = x[:, 0]

        # Classification head
        x = self.head(x)

        return x

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)


class ECC_ViT_Large(nn.Module):
    """
    Larger ViT with more layers and wider embedding.
    """

    def __init__(self, args, dropout=0.1, label_smoothing=0.0):
        super(ECC_ViT_Large, self).__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        self.n_z = code.H_z.shape[0]
        self.n_x = code.H_x.shape[0]
        self.grid_size = self.L

        # Compute stabilizer coordinates
        H_z_np = code.H_z.cpu().numpy() if torch.is_tensor(code.H_z) else code.H_z
        H_x_np = code.H_x.cpu().numpy() if torch.is_tensor(code.H_x) else code.H_x
        self.z_coord_map = compute_stabilizer_positions_from_H(H_z_np, self.L)
        self.x_coord_map = compute_stabilizer_positions_from_H(H_x_np, self.L)

        # Larger config
        d_model = getattr(args, 'd_model', 256)
        n_heads = getattr(args, 'h', 8)
        n_layers = getattr(args, 'N_dec', 8)

        self.d_model = d_model
        self.n_patches = self.L * self.L

        # Deeper CNN Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=1),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.n_patches, d_model) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head with hidden layer
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4)
        )

        # Loss
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _syndrome_to_grid(self, syndrome):
        """
        Convert flat syndrome to 2-channel 2D grid.

        Value encoding: 0 = empty, -1 = no error, +1 = error detected
        """
        batch_size = syndrome.shape[0]
        device = syndrome.device
        h = w = self.grid_size

        z_grid = torch.zeros(batch_size, h, w, device=device)
        x_grid = torch.zeros(batch_size, h, w, device=device)

        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        # Convert syndrome: 0 -> -1, 1 -> +1
        s_z_encoded = s_z * 2 - 1
        s_x_encoded = s_x * 2 - 1

        for idx, (row, col) in self.z_coord_map.items():
            if idx < self.n_z:
                z_grid[:, row, col] = s_z_encoded[:, idx]

        for idx, (row, col) in self.x_coord_map.items():
            if idx < self.n_x:
                x_grid[:, row, col] = s_x_encoded[:, idx]

        return torch.stack([z_grid, x_grid], dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        # Syndrome to 2D grid
        x = self._syndrome_to_grid(x)  # (B, 2, L, L)

        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, L, L)

        # Flatten patches to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, L*L, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+L*L, d_model)

        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Use CLS token for classification
        x = x[:, 0]

        # Classification head
        x = self.head(x)

        return x

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)


class ECC_ViT_QubitCentric(nn.Module):
    """
    ViT with QubitCentric input representation.

    Uses H.T @ syndrome to create dense qubit-level grid,
    then applies ViT architecture for global attention.

    Input: [H_z.T @ s_z, H_x.T @ s_x] - 2 channel qubit count grid
    """

    def __init__(self, args, dropout=0.1, label_smoothing=0.0):
        super(ECC_ViT_QubitCentric, self).__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        self.n_z = code.H_z.shape[0]
        self.n_x = code.H_x.shape[0]
        self.n_qubits = code.H_z.shape[1]

        # Store H matrices
        H_z_np = code.H_z.cpu().numpy() if torch.is_tensor(code.H_z) else code.H_z
        H_x_np = code.H_x.cpu().numpy() if torch.is_tensor(code.H_x) else code.H_x

        self.register_buffer('H_z', torch.from_numpy(H_z_np).float())
        self.register_buffer('H_x', torch.from_numpy(H_x_np).float())

        # Transformer config
        d_model = getattr(args, 'd_model', 128)
        n_heads = getattr(args, 'h', 8)
        n_layers = getattr(args, 'N_dec', 6)

        self.d_model = d_model
        self.n_patches = self.L * self.L

        # CNN Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=1),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.n_patches, d_model) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4)
        )

        # Loss
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _compute_qubit_grid(self, syndrome):
        """Compute 2-channel qubit count grid using H.T @ syndrome."""
        batch_size = syndrome.shape[0]
        L = self.L

        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        # H.T @ syndrome = qubit-centric count
        z_count = torch.matmul(s_z, self.H_z)  # (B, n_qubits)
        x_count = torch.matmul(s_x, self.H_x)  # (B, n_qubits)

        # Normalize to [0, 1] range (max count is 4 for surface code)
        z_count = z_count / 4.0
        x_count = x_count / 4.0

        z_grid = z_count.view(batch_size, L, L)
        x_grid = x_count.view(batch_size, L, L)

        return torch.stack([z_grid, x_grid], dim=1)  # (B, 2, L, L)

    def forward(self, x):
        batch_size = x.shape[0]

        # QubitCentric grid representation
        x = self._compute_qubit_grid(x)  # (B, 2, L, L)

        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, L, L)

        # Flatten patches to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, L*L, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+L*L, d_model)

        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Use CLS token for classification
        x = x[:, 0]

        # Classification head
        x = self.head(x)

        return x

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)


class ECC_ViT_LUT_Concat(nn.Module):
    """
    ViT with LUT Concat input (4 channels) - ALL IN QUBIT SPACE.

    Channel 0: Z-syndrome involvement (ternary: 0=empty, -1=ON, +1=OFF)
    Channel 1: X-syndrome involvement (ternary: 0=empty, -1=ON, +1=OFF)
    Channel 2: LUT Z-error prediction (ternary: 0=empty, -1=no error, +1=error)
    Channel 3: LUT X-error prediction (ternary: 0=empty, -1=no error, +1=error)
    """

    def __init__(self, args, x_error_lut, z_error_lut, dropout=0.1, label_smoothing=0.0):
        super(ECC_ViT_LUT_Concat, self).__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        self.n_z = code.H_z.shape[0]
        self.n_x = code.H_x.shape[0]
        self.n_qubits = code.H_z.shape[1]

        # Qubit 좌표 매핑
        self.qubit_coord_map = {}
        for q in range(self.n_qubits):
            r, c = q // self.L, q % self.L
            self.qubit_coord_map[q] = (r, c)

        # Convert LUT dict to tensor
        x_lut_tensor = torch.zeros(self.n_z, self.n_qubits)
        for i, err in x_error_lut.items():
            if i < self.n_z:
                x_lut_tensor[i] = err.float()
        self.register_buffer('x_lut', x_lut_tensor)

        z_lut_tensor = torch.zeros(self.n_x, self.n_qubits)
        for i, err in z_error_lut.items():
            if i < self.n_x:
                z_lut_tensor[i] = err.float()
        self.register_buffer('z_lut', z_lut_tensor)

        # Store H matrices
        H_z_np = code.H_z.cpu().numpy() if torch.is_tensor(code.H_z) else code.H_z
        H_x_np = code.H_x.cpu().numpy() if torch.is_tensor(code.H_x) else code.H_x

        self.register_buffer('H_z', torch.from_numpy(H_z_np).float())
        self.register_buffer('H_x', torch.from_numpy(H_x_np).float())

        # Transformer config
        d_model = getattr(args, 'd_model', 128)
        n_heads = getattr(args, 'h', 8)
        n_layers = getattr(args, 'N_dec', 6)

        self.d_model = d_model
        self.n_patches = self.L * self.L

        # CNN Patch Embedding (4 channels)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=1),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.n_patches, d_model) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4)
        )

        # Loss
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _batch_lut_lookup(self, syndrome):
        """Vectorized batch LUT lookup."""
        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        c_x = torch.matmul(s_z, self.x_lut) % 2
        c_z = torch.matmul(s_x, self.z_lut) % 2

        return c_z, c_x

    def _compute_concat_grid(self, syndrome):
        """
        Compute 4-channel grid in QUBIT SPACE with proper coordinate mapping.
        
        Returns: (B, 4, L, L)
        - Channel 0: Z-syndrome involvement (ternary: 0=empty, -1=ON, +1=OFF)
        - Channel 1: X-syndrome involvement (ternary: 0=empty, -1=ON, +1=OFF)
        - Channel 2: LUT Z-error (ternary: 0=empty, -1=no error, +1=error)
        - Channel 3: LUT X-error (ternary: 0=empty, -1=no error, +1=error)
        """
        batch_size = syndrome.shape[0]
        L = self.L
        device = syndrome.device

        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        # -----------------------------------------------------------
        # Step 1: Project syndrome to qubit space via H^T
        # -----------------------------------------------------------
        # Result: count of syndromes involved (0~4 range)
        real_z_count = torch.matmul(s_z, self.H_z)  # (B, n_qubits)
        real_x_count = torch.matmul(s_x, self.H_x)

        # -----------------------------------------------------------
        # [NEW] Convert to ternary: 켜짐(count>0) → -1, 꺼짐(count=0) → +1
        # -----------------------------------------------------------
        # mask: True where syndrome is ON (count > 0)
        z_syndrome_on = (real_z_count > 0).float()  # 1 if ON, 0 if OFF
        x_syndrome_on = (real_x_count > 0).float()
        
        # Convert: ON(1) → -1, OFF(0) → +1
        # Formula: -2 * mask + 1
        real_z_ternary = -2 * z_syndrome_on + 1  # ON: -2*1+1=-1, OFF: -2*0+1=+1
        real_x_ternary = -2 * x_syndrome_on + 1

        # ----------------------------------------------------------
        # Step 2: LUT lookup
        # -----------------------------------------------------------
        lut_e_z, lut_e_x = self._batch_lut_lookup(syndrome)  # (B, n_qubits), binary {0,1}

        # Convert to ternary: 0 → -1, 1 → +1
        lut_e_z_ternary = lut_e_z * 2 - 1  # {-1, +1}
        lut_e_x_ternary = lut_e_x * 2 - 1

        # -----------------------------------------------------------
        # Step 3: Initialize grids with 0 (empty positions)
        # -----------------------------------------------------------
        real_z_grid = torch.zeros(batch_size, L, L, device=device)
        real_x_grid = torch.zeros(batch_size, L, L, device=device)
        lut_z_grid = torch.zeros(batch_size, L, L, device=device)
        lut_x_grid = torch.zeros(batch_size, L, L, device=device)

        # -----------------------------------------------------------
        # Step 4: Scatter to actual qubit positions
        # -----------------------------------------------------------
        for q_idx, (r, c) in self.qubit_coord_map.items():
            if q_idx < self.n_qubits and r < L and c < L:
                real_z_grid[:, r, c] = real_z_ternary[:, q_idx]
                real_x_grid[:, r, c] = real_x_ternary[:, q_idx]
                lut_z_grid[:, r, c] = lut_e_z_ternary[:, q_idx]
                lut_x_grid[:, r, c] = lut_e_x_ternary[:, q_idx]

        # Stack into 4-channel tensor
        return torch.stack([real_z_grid, real_x_grid, lut_z_grid, lut_x_grid], dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        # 4-channel grid representation
        x = self._compute_concat_grid(x)  # (B, 4, L, L)

        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, L, L)

        # Flatten patches to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, L*L, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+L*L, d_model)

        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Use CLS token for classification
        x = x[:, 0]

        # Classification head
        x = self.head(x)

        return x

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)
    


class ECC_Transformer(nn.Module):
    """
    Standard Transformer decoder for ECC.

    Flow:
    Syndrome (flat) -> Linear Embedding -> Transformer -> Classifier
    """

    def __init__(self, args, dropout=0.1, label_smoothing=0.0):
        super(ECC_Transformer, self).__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        self.n_z = code.H_z.shape[0]
        self.n_x = code.H_x.shape[0]
        self.n_syndromes = self.n_z + self.n_x

        # Transformer config
        d_model = getattr(args, 'd_model', 128)
        n_heads = getattr(args, 'h', 8)
        n_layers = getattr(args, 'N_dec', 6)

        self.d_model = d_model

        # Linear embedding for syndrome
        self.syndrome_embed = nn.Linear(1, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.n_syndromes, d_model) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4)
        )

        # Loss
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size = x.shape[0]

        # Encode syndrome: 0 -> -1, 1 -> 1
        x = x * 2 - 1.0

        # Linear embedding
        x = x.unsqueeze(-1)  # (B, n_syndromes, 1)
        x = self.syndrome_embed(x)  # (B, n_syndromes, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + n_syndromes, d_model)

        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Use CLS token for classification
        x = x[:, 0]

        # Classification head
        x = self.head(x)

        return x

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)
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
    
    [LOGIC UPDATE]
    - Base Value: 0.0 (Clean / Empty)
    - Signal Value: > 0.0 (Error / Syndrome Involvement)
    - Mapping: Physical Coordinates -> Dense Indices (0~d-1)
    - Normalization: Divided by 2.0 (Max connectivity per type is 2)
    
    Channels:
    0: Z-syndrome count (0.0, 0.5, 1.0)
    1: X-syndrome count (0.0, 0.5, 1.0)
    2: LUT Z-error prediction (Binary 0.0 or 1.0)
    3: LUT X-error prediction (Binary 0.0 or 1.0)
    """

    def __init__(self, args, x_error_lut, z_error_lut, dropout=0.1, label_smoothing=0.0):
        super(ECC_ViT_LUT_Concat, self).__init__()
        self.args = args
        
        code = args.code
        self.n_z = code.H_z.shape[0]
        self.n_x = code.H_x.shape[0]
        self.n_qubits = code.H_z.shape[1]

        # -----------------------------------------------------------
        # [CHANGE 1] Dense Grid Size Calculation
        # -----------------------------------------------------------
        self.L = int(np.sqrt(self.n_qubits))
        
        # -----------------------------------------------------------
        # [CHANGE 2] Physical Coords -> Dense Index Mapping
        # -----------------------------------------------------------
        if hasattr(code, 'qubit_coordinates'):
            coords = code.qubit_coordinates 
        else:
            print("Warning: No coordinates found. Using linear mapping.")
            coords = [(i // self.L, i % self.L) for i in range(self.n_qubits)]

        xs = torch.tensor([c[0] for c in coords])
        ys = torch.tensor([c[1] for c in coords])

        unique_x = torch.unique(xs, sorted=True)
        unique_y = torch.unique(ys, sorted=True)

        qubit_rows = torch.bucketize(xs, unique_x)
        qubit_cols = torch.bucketize(ys, unique_y)

        self.register_buffer('qubit_rows', qubit_rows)
        self.register_buffer('qubit_cols', qubit_cols)

        # -----------------------------------------------------------
        # LUT & Matrices Setup
        # -----------------------------------------------------------
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

        H_z_np = code.H_z.cpu().numpy() if torch.is_tensor(code.H_z) else code.H_z
        H_x_np = code.H_x.cpu().numpy() if torch.is_tensor(code.H_x) else code.H_x

        self.register_buffer('H_z', torch.from_numpy(H_z_np).float())
        self.register_buffer('H_x', torch.from_numpy(H_x_np).float())

        # -----------------------------------------------------------
        # Transformer Components
        # -----------------------------------------------------------
        d_model = getattr(args, 'd_model', 128)
        n_heads = getattr(args, 'h', 8)
        n_layers = getattr(args, 'N_dec', 6)

        self.d_model = d_model
        self.n_patches = self.L * self.L

        self.patch_embed = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=1),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.n_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4)
        )

        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _batch_lut_lookup(self, syndrome):
        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]
        c_x = torch.matmul(s_z, self.x_lut) % 2
        c_z = torch.matmul(s_x, self.z_lut) % 2
        return c_z, c_x

    def _compute_concat_grid(self, syndrome):
        batch_size = syndrome.shape[0]
        L = self.L
        device = syndrome.device

        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        # -----------------------------------------------------------
        # Step 1: Syndrome Count (0.0 ~ 1.0)
        # -----------------------------------------------------------
        real_z_count = torch.matmul(s_z, self.H_z) 
        real_x_count = torch.matmul(s_x, self.H_x)

        # [CHANGE 3] Normalize by 2.0 instead of 4.0
        # Since channels are separated (Z/X), max connectivity is 2.
        real_z_norm = real_z_count / 2.0
        real_x_norm = real_x_count / 2.0

        # -----------------------------------------------------------
        # Step 2: LUT Prediction (0.0 or 1.0)
        # -----------------------------------------------------------
        lut_e_z, lut_e_x = self._batch_lut_lookup(syndrome)
        lut_e_z = lut_e_z.float()
        lut_e_x = lut_e_x.float()

        # -----------------------------------------------------------
        # Step 3: Initialize Grid & Scatter
        # -----------------------------------------------------------
        grid_stack = torch.zeros(batch_size, 4, L, L, device=device)

        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.n_qubits)
        rows = self.qubit_rows.unsqueeze(0).expand(batch_size, -1)
        cols = self.qubit_cols.unsqueeze(0).expand(batch_size, -1)
        
        grid_stack[batch_idx, 0, rows, cols] = real_z_norm
        grid_stack[batch_idx, 1, rows, cols] = real_x_norm
        grid_stack[batch_idx, 2, rows, cols] = lut_e_z
        grid_stack[batch_idx, 3, rows, cols] = lut_e_x

        return grid_stack

    def forward(self, x):
        batch_size = x.shape[0]
        x = self._compute_concat_grid(x) 
        x = self.patch_embed(x) 
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) 
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 0]
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
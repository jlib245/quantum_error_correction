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
        """Convert flat syndrome to 2-channel 2D grid."""
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
        """Convert flat syndrome to 2-channel 2D grid."""
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

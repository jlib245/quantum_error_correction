"""
Diamond CNN Decoder for Rotated Surface Code

45도 회전된 grid에서 2x2 conv로 Q-Z-X-Q 마름모 패턴을 캡처.

Grid 변환:
- Original (2L+1) x (2L+1) sparse grid (빈칸 많음)
- Rotated ~(2L-1) x (2L-1) dense grid (빈칸 최소화)

2x2 conv가 자연스럽게 [Q, X; Z, Q] 또는 [Q, Z; X, Q] 패턴을 캡처.

Channels:
- Ch0: LUT predicted X errors at Q positions
- Ch1: LUT predicted Z errors at Q positions
- Ch2: Z syndrome at Z positions
- Ch3: X syndrome at X positions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiamondGridBuilder(nn.Module):
    """
    Syndrome + LUT prediction을 45도 회전된 dense grid로 변환.

    6 Channels:
    - Ch0: LUT predicted X errors (from Z syndrome) at Q positions
    - Ch1: LUT predicted Z errors (from X syndrome) at Q positions
    - Ch2: Z syndrome at Z stabilizer positions
    - Ch3: X syndrome at X stabilizer positions
    - Ch4: H_z.T @ s_z (aggregated Z syndrome count) at Q positions
    - Ch5: H_x.T @ s_x (aggregated X syndrome count) at Q positions

    Value encoding for syndrome channels (Ch2, Ch3):
    - 0 = empty (no stabilizer at this position)
    - -1 = stabilizer present, no error detected
    - +1 = stabilizer present, error detected

    Value encoding for LUT channels (Ch0, Ch1):
    - 0 = empty or no error predicted
    - 1 = error predicted

    Value encoding for count channels (Ch4, Ch5):
    - 0 = empty or zero count
    - 0~1 = normalized count (0~4) / 4
    """

    def __init__(self, L, H_z, H_x, x_error_lut=None, z_error_lut=None):
        super().__init__()
        self.L = L
        self.n_qubits = L * L
        self.old_size = 2 * L + 1
        self.new_size = 2 * L - 1

        # H matrices 저장
        if isinstance(H_z, np.ndarray):
            H_z = torch.from_numpy(H_z).float()
        if isinstance(H_x, np.ndarray):
            H_x = torch.from_numpy(H_x).float()

        self.register_buffer('H_z', H_z)
        self.register_buffer('H_x', H_x)

        self.n_z = H_z.shape[0]
        self.n_x = H_x.shape[0]

        # LUT를 tensor로 변환 (vectorized lookup용)
        if x_error_lut is not None:
            x_lut_tensor = torch.zeros(self.n_z, self.n_qubits)
            for i, err in x_error_lut.items():
                if i < self.n_z:
                    x_lut_tensor[i] = err.float()
            self.register_buffer('x_lut', x_lut_tensor)
        else:
            self.register_buffer('x_lut', None)

        if z_error_lut is not None:
            z_lut_tensor = torch.zeros(self.n_x, self.n_qubits)
            for i, err in z_error_lut.items():
                if i < self.n_x:
                    z_lut_tensor[i] = err.float()
            self.register_buffer('z_lut', z_lut_tensor)
        else:
            self.register_buffer('z_lut', None)

        # 좌표 매핑 미리 계산
        self._precompute_mappings()

    def _precompute_mappings(self):
        """좌표 변환 매핑 미리 계산 (vectorized indexing용)"""
        L = self.L

        # Qubit positions: original (2i+1, 2j+1) for i,j in [0, L-1]
        qubit_rows = []
        qubit_cols = []
        qubit_src_indices = []
        for i in range(L):
            for j in range(L):
                old_r, old_c = 2*i + 1, 2*j + 1
                new_r, new_c = self._rotate_45(old_r, old_c)
                if new_r < self.new_size and new_c < self.new_size:
                    qubit_rows.append(new_r)
                    qubit_cols.append(new_c)
                    qubit_src_indices.append(i * L + j)

        # Z-stabilizer positions (from H_z)
        z_stab_rows = []
        z_stab_cols = []
        z_stab_src_indices = []
        for stab_idx in range(self.n_z):
            qubits = torch.where(self.H_z[stab_idx] == 1)[0].tolist()
            old_r, old_c = self._get_stabilizer_position(qubits, 'Z')
            new_r, new_c = self._rotate_45(old_r, old_c)
            if new_r < self.new_size and new_c < self.new_size:
                z_stab_rows.append(new_r)
                z_stab_cols.append(new_c)
                z_stab_src_indices.append(stab_idx)

        # X-stabilizer positions (from H_x)
        x_stab_rows = []
        x_stab_cols = []
        x_stab_src_indices = []
        for stab_idx in range(self.n_x):
            qubits = torch.where(self.H_x[stab_idx] == 1)[0].tolist()
            old_r, old_c = self._get_stabilizer_position(qubits, 'X')
            new_r, new_c = self._rotate_45(old_r, old_c)
            if new_r < self.new_size and new_c < self.new_size:
                x_stab_rows.append(new_r)
                x_stab_cols.append(new_c)
                x_stab_src_indices.append(stab_idx)

        # Register as buffers for vectorized indexing
        self.register_buffer('qubit_rows', torch.tensor(qubit_rows, dtype=torch.long))
        self.register_buffer('qubit_cols', torch.tensor(qubit_cols, dtype=torch.long))
        self.register_buffer('qubit_src_idx', torch.tensor(qubit_src_indices, dtype=torch.long))

        self.register_buffer('z_stab_rows', torch.tensor(z_stab_rows, dtype=torch.long))
        self.register_buffer('z_stab_cols', torch.tensor(z_stab_cols, dtype=torch.long))
        self.register_buffer('z_stab_src_idx', torch.tensor(z_stab_src_indices, dtype=torch.long))

        self.register_buffer('x_stab_rows', torch.tensor(x_stab_rows, dtype=torch.long))
        self.register_buffer('x_stab_cols', torch.tensor(x_stab_cols, dtype=torch.long))
        self.register_buffer('x_stab_src_idx', torch.tensor(x_stab_src_indices, dtype=torch.long))

    def _get_stabilizer_position(self, qubits, stab_type):
        """Stabilizer의 original grid 위치 계산"""
        L = self.L
        rows = [q // L for q in qubits]
        cols = [q % L for q in qubits]

        if len(qubits) == 4:  # Interior stabilizer
            old_r = 2 * min(rows) + 2
            old_c = 2 * min(cols) + 2
        else:  # Boundary stabilizer (weight-2)
            if stab_type == 'Z':
                # Z boundary: left/right edges
                if min(cols) == 0 and max(cols) == 0:  # Left
                    old_c = 0
                    old_r = 2 * min(rows) + 2
                else:  # Right
                    old_c = 2 * L
                    old_r = 2 * min(rows) + 2
            else:  # X boundary: top/bottom edges
                if min(rows) == 0 and max(rows) == 0:  # Top
                    old_r = 0
                    old_c = 2 * min(cols) + 2
                else:  # Bottom
                    old_r = 2 * L
                    old_c = 2 * min(cols) + 2

        return old_r, old_c

    def _rotate_45(self, old_r, old_c):
        """45도 회전 좌표 변환"""
        L = self.L
        new_r = L - 1 + (old_r - old_c) // 2
        new_c = (old_r + old_c) // 2 - 1
        # Boundary 조정
        new_r = max(0, new_r)
        new_c = max(0, new_c)
        return new_r, new_c

    def forward(self, syndrome):
        """
        Syndrome을 45도 회전된 grid로 변환 (LUT prediction + aggregated syndrome 포함).
        Vectorized implementation for speed.

        Args:
            syndrome: (B, n_z + n_x) - [s_z, s_x] concatenated

        Returns:
            grid: (B, 6, H, W) - 6 channels:
                Ch0: LUT predicted X errors at Q positions
                Ch1: LUT predicted Z errors at Q positions
                Ch2: Z syndrome at Z positions (0=empty, -1=no error, +1=error)
                Ch3: X syndrome at X positions (0=empty, -1=no error, +1=error)
                Ch4: H_z.T @ s_z (aggregated Z syndrome count) at Q positions
                Ch5: H_x.T @ s_x (aggregated X syndrome count) at Q positions
        """
        B = syndrome.shape[0]
        device = syndrome.device

        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        # LUT predictions (XOR via matmul + mod 2)
        if self.x_lut is not None:
            lut_x_error = torch.matmul(s_z, self.x_lut) % 2  # (B, n_qubits)
        else:
            lut_x_error = torch.zeros(B, self.n_qubits, device=device)

        if self.z_lut is not None:
            lut_z_error = torch.matmul(s_x, self.z_lut) % 2  # (B, n_qubits)
        else:
            lut_z_error = torch.zeros(B, self.n_qubits, device=device)

        # Aggregated syndrome counts: H.T @ syndrome (how many stabilizers point to each qubit)
        # Normalized to 0~1 range (max 4 stabilizers can point to a qubit)
        z_count = torch.matmul(s_z, self.H_z) / 4.0  # (B, n_qubits)
        x_count = torch.matmul(s_x, self.H_x) / 4.0  # (B, n_qubits)

        # Initialize rotated grid with 0 (empty)
        grid = torch.zeros((B, 6, self.new_size, self.new_size), device=device)

        # Vectorized assignment using advanced indexing
        # Channel 0: LUT predicted X errors at Q positions (0/1)
        grid[:, 0, self.qubit_rows, self.qubit_cols] = lut_x_error[:, self.qubit_src_idx]

        # Channel 1: LUT predicted Z errors at Q positions (0/1)
        grid[:, 1, self.qubit_rows, self.qubit_cols] = lut_z_error[:, self.qubit_src_idx]

        # Channel 2: Z-syndrome values (0 -> -1, 1 -> +1)
        s_z_encoded = s_z * 2 - 1  # Convert: 0 -> -1, 1 -> +1
        grid[:, 2, self.z_stab_rows, self.z_stab_cols] = s_z_encoded[:, self.z_stab_src_idx]

        # Channel 3: X-syndrome values (0 -> -1, 1 -> +1)
        s_x_encoded = s_x * 2 - 1  # Convert: 0 -> -1, 1 -> +1
        grid[:, 3, self.x_stab_rows, self.x_stab_cols] = s_x_encoded[:, self.x_stab_src_idx]

        # Channel 4: Aggregated Z syndrome count at Q positions
        grid[:, 4, self.qubit_rows, self.qubit_cols] = z_count[:, self.qubit_src_idx]

        # Channel 5: Aggregated X syndrome count at Q positions
        grid[:, 5, self.qubit_rows, self.qubit_cols] = x_count[:, self.qubit_src_idx]

        return grid


class ECC_DiamondCNN(nn.Module):
    """
    Diamond CNN Decoder with LUT predictions.

    45도 회전된 grid에서 2x2 conv로 Q-Z-X-Q 패턴 캡처.
    LUT prediction을 qubit 채널에 배치하여 도메인 지식 활용.
    """

    def __init__(self, args, x_error_lut=None, z_error_lut=None,
                 dropout=0.1, label_smoothing=0.0):
        super().__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        H_z = code.H_z.cpu().numpy() if torch.is_tensor(code.H_z) else code.H_z
        H_x = code.H_x.cpu().numpy() if torch.is_tensor(code.H_x) else code.H_x

        # Grid builder with LUT
        self.grid_builder = DiamondGridBuilder(
            self.L, H_z, H_x,
            x_error_lut=x_error_lut,
            z_error_lut=z_error_lut
        )

        d_model = getattr(args, 'd_model', 128)

        # CNN with 2x2 conv for diamond pattern (6 channels input)
        self.conv1 = nn.Conv2d(6, 64, kernel_size=2, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, d_model, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(d_model)

        # Global pooling + classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
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

    def forward(self, syndrome):
        # Build rotated grid with LUT predictions
        x = self.grid_builder(syndrome)  # (B, 4, H, W)

        # CNN
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))

        # Classifier
        x = self.head(x)
        return x

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)


class ECC_DiamondCNN_Deep(nn.Module):
    """
    Deeper Diamond CNN with residual connections and LUT predictions.
    """

    def __init__(self, args, x_error_lut=None, z_error_lut=None,
                 dropout=0.1, label_smoothing=0.0):
        super().__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        H_z = code.H_z.cpu().numpy() if torch.is_tensor(code.H_z) else code.H_z
        H_x = code.H_x.cpu().numpy() if torch.is_tensor(code.H_x) else code.H_x

        # Grid builder with LUT
        self.grid_builder = DiamondGridBuilder(
            self.L, H_z, H_x,
            x_error_lut=x_error_lut,
            z_error_lut=z_error_lut
        )

        d_model = getattr(args, 'd_model', 128)

        # Initial 2x2 conv for diamond pattern (6 channels input)
        self.stem = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Residual blocks with 3x3 conv
        self.res_blocks = nn.ModuleList([
            self._make_res_block(64, 64),
            self._make_res_block(64, 128),
            self._make_res_block(128, d_model),
        ])

        # Global pooling + classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
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

    def _make_res_block(self, in_ch, out_ch):
        return nn.ModuleDict({
            'conv1': nn.Conv2d(in_ch, out_ch, 3, padding=1),
            'bn1': nn.BatchNorm2d(out_ch),
            'conv2': nn.Conv2d(out_ch, out_ch, 3, padding=1),
            'bn2': nn.BatchNorm2d(out_ch),
            'shortcut': nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        })

    def _apply_res_block(self, x, block):
        identity = block['shortcut'](x)
        out = F.gelu(block['bn1'](block['conv1'](x)))
        out = block['bn2'](block['conv2'](out))
        return F.gelu(out + identity)

    def forward(self, syndrome):
        x = self.grid_builder(syndrome)
        x = self.stem(x)

        for block in self.res_blocks:
            x = self._apply_res_block(x, block)

        return self.head(x)

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)


class SpatialAttention(nn.Module):
    """Spatial self-attention for 2D feature maps."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, heads, head_dim, HW)

        # Attention
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        k = k.permute(0, 1, 2, 3)  # (B, heads, head_dim, HW)
        v = v.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)

        attn = (q @ k) * self.scale  # (B, heads, HW, HW)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return self.proj(out)


class DilatedResBlock(nn.Module):
    """Residual block with dilated convolution for larger receptive field."""

    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + identity)


class ECC_DiamondCNN_Attention(nn.Module):
    """
    Diamond CNN with Dilated Convolutions + Self-Attention.

    Key improvements:
    1. 2x2 conv captures local diamond pattern (Q-Z-X-Q)
    2. Dilated convs expand receptive field for error chains
    3. Self-attention captures global syndrome correlations
    4. Multi-scale feature fusion
    """

    def __init__(self, args, x_error_lut=None, z_error_lut=None,
                 dropout=0.1, label_smoothing=0.0):
        super().__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        H_z = code.H_z.cpu().numpy() if torch.is_tensor(code.H_z) else code.H_z
        H_x = code.H_x.cpu().numpy() if torch.is_tensor(code.H_x) else code.H_x

        # Grid builder with LUT
        self.grid_builder = DiamondGridBuilder(
            self.L, H_z, H_x,
            x_error_lut=x_error_lut,
            z_error_lut=z_error_lut
        )

        d_model = getattr(args, 'd_model', 128)

        # Stage 1: Local diamond pattern extraction (2x2 conv, 6 channels input)
        self.stem = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Stage 2: Multi-scale dilated convolutions
        self.dilated_blocks = nn.ModuleList([
            DilatedResBlock(64, 64, dilation=1),    # RF: 3x3
            DilatedResBlock(64, 128, dilation=2),   # RF: 7x7
            DilatedResBlock(128, 128, dilation=4),  # RF: 15x15 (covers full grid)
        ])

        # Stage 3: Self-attention for global correlation
        self.attn = SpatialAttention(128, num_heads=4)
        self.attn_norm = nn.BatchNorm2d(128)

        # Stage 4: Final projection
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        # Classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
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

    def forward(self, syndrome):
        # Build rotated grid with LUT predictions
        x = self.grid_builder(syndrome)  # (B, 4, H, W)

        # Local pattern extraction
        x = self.stem(x)

        # Multi-scale dilated convolutions
        for block in self.dilated_blocks:
            x = block(x)

        # Global attention
        x = x + self.attn(self.attn_norm(x))

        # Final projection
        x = self.final_conv(x)

        # Classifier
        return self.head(x)

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)

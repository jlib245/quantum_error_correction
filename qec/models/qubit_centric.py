"""
Qubit-Centric CNN Decoder for Quantum Error Correction

Uses H.T @ syndrome as qubit-centric representation.
Optionally combines with LUT predictions.

Models:
- ECC_QubitCentric: H.T @ syndrome only (2ch)
- ECC_LUT_Residual: residual = real_count - lut_count (2ch)
- ECC_LUT_Concat: [real_count, lut_error] (4ch)
"""
import torch
import torch.nn as nn


class ECC_QubitCentric(nn.Module):
    """
    Qubit-centric CNN decoder (no LUT).

    Input: [H_z.T @ s_z, H_x.T @ s_x] - 2 channel qubit count grid
    """

    def __init__(self, args, dropout=0.1, label_smoothing=0.0):
        super(ECC_QubitCentric, self).__init__()
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

        # CNN backbone (2 channels)
        d_model = getattr(args, 'd_model', 128)

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

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

    def _compute_qubit_grid(self, syndrome):
        """Compute 2-channel qubit count grid."""
        batch_size = syndrome.shape[0]
        L = self.L

        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        z_count = torch.matmul(s_z, self.H_z)
        x_count = torch.matmul(s_x, self.H_x)

        z_grid = z_count.view(batch_size, L, L)
        x_grid = x_count.view(batch_size, L, L)

        return torch.stack([z_grid, x_grid], dim=1)

    def forward(self, syndrome):
        x = self._compute_qubit_grid(syndrome)
        x = self.cnn(x)
        x = self.head(x)
        return x

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)


class ECC_LUT_Residual(nn.Module):
    """
    LUT Residual CNN decoder.

    Input: residual = (H.T @ syndrome) - (H.T @ H @ lut_error)
    Model learns what LUT couldn't explain.
    """

    def __init__(self, args, x_error_lut, z_error_lut, dropout=0.1, label_smoothing=0.0):
        super(ECC_LUT_Residual, self).__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        self.n_z = code.H_z.shape[0]
        self.n_x = code.H_x.shape[0]
        self.n_qubits = code.H_z.shape[1]

        # Convert LUT dict to tensor for vectorized lookup
        # x_error_lut[i] = error pattern for z-stabilizer i
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

        # Precompute H.T @ H
        self.register_buffer('HtH_z', self.H_z.T @ self.H_z)
        self.register_buffer('HtH_x', self.H_x.T @ self.H_x)

        # CNN backbone (2 channels)
        d_model = getattr(args, 'd_model', 128)

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

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

    def _batch_lut_lookup(self, syndrome):
        """Vectorized batch LUT lookup using matmul."""
        # s_z: (B, n_z), s_x: (B, n_x)
        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        # Matmul gives weighted sum, then mod 2 for XOR effect
        # c_x = (s_z @ x_lut) % 2  -> X errors from Z stabilizers
        # c_z = (s_x @ z_lut) % 2  -> Z errors from X stabilizers
        c_x = torch.matmul(s_z, self.x_lut) % 2
        c_z = torch.matmul(s_x, self.z_lut) % 2

        return c_z, c_x  # (B, n_qubits) each

    def _compute_residual_grid(self, syndrome):
        """Compute residual grid."""
        batch_size = syndrome.shape[0]
        L = self.L

        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        # Real qubit count
        real_z_count = torch.matmul(s_z, self.H_z)
        real_x_count = torch.matmul(s_x, self.H_x)

        # LUT lookup
        lut_e_z, lut_e_x = self._batch_lut_lookup(syndrome)

        # LUT qubit count: H.T @ H @ lut_error
        lut_z_count = torch.matmul(lut_e_x, self.HtH_z)
        lut_x_count = torch.matmul(lut_e_z, self.HtH_x)

        # Residual
        residual_z = real_z_count - lut_z_count
        residual_x = real_x_count - lut_x_count

        residual_z_grid = residual_z.view(batch_size, L, L)
        residual_x_grid = residual_x.view(batch_size, L, L)

        return torch.stack([residual_z_grid, residual_x_grid], dim=1)

    def forward(self, syndrome):
        x = self._compute_residual_grid(syndrome)
        x = self.cnn(x)
        x = self.head(x)
        return x

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)


class ECC_LUT_Concat(nn.Module):
    """
    LUT Concat CNN decoder.

    Input: [real_z_count, real_x_count, lut_z_error, lut_x_error] (4ch)
    """

    def __init__(self, args, x_error_lut, z_error_lut, dropout=0.1, label_smoothing=0.0):
        super(ECC_LUT_Concat, self).__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        self.n_z = code.H_z.shape[0]
        self.n_x = code.H_x.shape[0]
        self.n_qubits = code.H_z.shape[1]

        # Convert LUT dict to tensor for vectorized lookup
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

        # CNN backbone (4 channels)
        d_model = getattr(args, 'd_model', 128)

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

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

    def _batch_lut_lookup(self, syndrome):
        """Vectorized batch LUT lookup using matmul."""
        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        c_x = torch.matmul(s_z, self.x_lut) % 2
        c_z = torch.matmul(s_x, self.z_lut) % 2

        return c_z, c_x

    def _compute_concat_grid(self, syndrome):
        """Compute 4-channel grid."""
        batch_size = syndrome.shape[0]
        L = self.L

        s_z = syndrome[:, :self.n_z]
        s_x = syndrome[:, self.n_z:]

        # Real qubit count
        real_z_count = torch.matmul(s_z, self.H_z)
        real_x_count = torch.matmul(s_x, self.H_x)

        # LUT lookup
        lut_e_z, lut_e_x = self._batch_lut_lookup(syndrome)

        # Reshape to grid
        real_z_grid = real_z_count.view(batch_size, L, L)
        real_x_grid = real_x_count.view(batch_size, L, L)
        lut_z_grid = lut_e_z.view(batch_size, L, L)
        lut_x_grid = lut_e_x.view(batch_size, L, L)

        return torch.stack([real_z_grid, real_x_grid, lut_z_grid, lut_x_grid], dim=1)

    def forward(self, syndrome):
        x = self._compute_concat_grid(syndrome)
        x = self.cnn(x)
        x = self.head(x)
        return x

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)

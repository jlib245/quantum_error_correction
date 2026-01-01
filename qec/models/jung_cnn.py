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
    Exact implementation including 'Stacking' for measurement error.
    
    Ref: "In Fig. 4(c), several measurements are stacked in the z-direction... with depth d+1."
    """
    def __init__(self, args, dropout=0.0, label_smoothing=0.0):
        super().__init__()
        self.args = args
        self.L = args.code_L

        code = args.code
        self.n_z = code.H_z.shape[0]
        self.n_x = code.H_x.shape[0]

        # Jung et al. uses (L+1)x(L+1) grid
        self.grid_size_h = self.L + 1
        self.grid_size_w = self.L + 1

        H_z_np = code.H_z.cpu().numpy() if torch.is_tensor(code.H_z) else code.H_z
        H_x_np = code.H_x.cpu().numpy() if torch.is_tensor(code.H_x) else code.H_x

        self.z_coord_map = compute_stabilizer_positions_from_H(H_z_np, self.L)
        self.x_coord_map = compute_stabilizer_positions_from_H(H_x_np, self.L)
        
        # -------------------------------------------------------------------
        # [수정 1] 논문대로 측정 오류 시 d+1 만큼 스택킹 (Fig 4c, )
        # -------------------------------------------------------------------
        if args.p_meas > 0:
            self.input_depth = self.L + 1 # Stack depth = d + 1
        else:
            self.input_depth = 1 # Code capacity model

        # 채널 수: (Z + X) * Depth
        in_channels = 2 * self.input_depth

        # Hyperparameter settings (Table 1 & Eq 3) [cite: 374, 412]
        if self.L == 3:
            n_filters = 8
        elif self.L == 5:
            n_filters = 32
        else: # L >= 7
            n_filters = 64
            
        # Layer 1
        # 논문은 Depth가 깊어져도 2D Conv를 사용하여 채널로 처리함
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()
        
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        
        # Flatten dimension calculation
        # Output spatial size is (L+2)x(L+2) due to padding logic in previous code
        self.flatten_dim = n_filters * (self.L + 2) * (self.L + 2)
        
        # Dense Layers [cite: 426]
        self.fc1 = nn.Linear(self.flatten_dim, 50)
        self.fc2 = nn.Linear(50, 4) # Output: I, X, Y, Z

        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _syndrome_to_grid(self, syndrome):
        """
        Convert syndrome to stacked 2D grids.
        
        Input syndrome shape: 
          - (Batch, n_syndromes) if p_meas=0
          - (Batch, Time, n_syndromes) if p_meas>0
        """
        batch_size = syndrome.shape[0]
        device = syndrome.device
        h, w = self.grid_size_h, self.grid_size_w
        
        # -------------------------------------------------------------------
        # [수정 2] 입력 데이터 Reshape (Time 축 처리)
        # -------------------------------------------------------------------
        # 만약 입력이 (Batch, Time*Syndrome)으로 Flat하게 들어왔다면 Reshape 필요
        if self.input_depth > 1:
            if syndrome.dim() == 2:
                # (Batch, (d+1)*n_syn) -> (Batch, d+1, n_syn)
                n_syn_total = self.n_z + self.n_x
                syndrome = syndrome.view(batch_size, self.input_depth, n_syn_total)
        else:
            if syndrome.dim() == 2:
                syndrome = syndrome.unsqueeze(1) # (Batch, 1, n_syn)

        grids = []

        # -------------------------------------------------------------------
        # [수정 3] Time Steps 만큼 반복하여 그리드 생성 후 스택킹
        # -------------------------------------------------------------------
        for t in range(self.input_depth):
            # t번째 타임스텝의 신드롬 추출
            s_curr = syndrome[:, t, :] # (Batch, n_syn)
            
            s_z = s_curr[:, :self.n_z]
            s_x = s_curr[:, self.n_z:]

            # 논문에서 제안한 Incoherent value m = -0.5 로 배경 초기화 
            z_grid = torch.full((batch_size, h, w), -0.5, device=device)
            x_grid = torch.full((batch_size, h, w), -0.5, device=device)

            # 신드롬 매핑 (값은 0 또는 1)
            # 논문: "syndromes... values are 0 and 1"
            
            for idx, (row, col) in self.z_coord_map.items():
                if idx < self.n_z:
                    # Valid range check
                    if row < h and col < w:
                        z_grid[:, row, col] = s_z[:, idx].float()

            for idx, (row, col) in self.x_coord_map.items():
                if idx < self.n_x:
                    if row < h and col < w:
                        x_grid[:, row, col] = s_x[:, idx].float()
            
            grids.append(z_grid)
            grids.append(x_grid)

        # 최종 스택킹: (Batch, 2*(d+1), H, W)
        # 순서: Z_t0, X_t0, Z_t1, X_t1, ...
        return torch.stack(grids, dim=1)

    def forward(self, x):
        # x shape expects: (Batch, Time, n_syndromes) or Flat version
        
        # 1. Grid 변환 (Stacking 포함)
        x = self._syndrome_to_grid(x) 
        
        # 2. CNN Layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # 3. Flatten & Dense
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)
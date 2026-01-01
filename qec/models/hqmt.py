import torch
import torch.nn as nn
import numpy as np
import copy

# 기존 transformer.py에서 필요한 모듈 재사용
from qec.models.transformer import (
    Encoder, 
    EncoderLayer, 
    MultiHeadedAttention, 
    PositionwiseFeedForward, 
    PositionalEncoding2D, 
    clones
)

class HQMT(nn.Module):
    """
    Hierarchical Qubit-Merging Transformer (HQMT)
    Reference: arXiv:2510.11593v1 [quant-ph]
    """
    def __init__(self, args, dropout=0.1, label_smoothing=0.0):
        super(HQMT, self).__init__()
        self.args = args
        self.d_model = args.d_model
        self.n_heads = args.h
        self.n_layers = args.N_dec  # 각 스테이지별 레이어 수 (논문 기본값 N=3)
        
        code = args.code
        self.n_qubits = code.n
        # 논문에서는 L x L 격자라고 가정 (Surface Code)
        self.L = args.code_L 

        # -------------------------------------------------------------------
        # 1. Adjacency Build (Qubit-Centric Patch Extraction 준비)
        # -------------------------------------------------------------------
        # 각 큐비트가 어떤 Z-stabilizer, X-stabilizer와 연결되어 있는지 인덱싱
        # Surface Code에서 큐비트는 최대 2개의 Z, 2개의 X 안정자와 연결됨 (총 4개)
        self.max_degree = 4 
        
        # (n_qubits, max_degree) 형태의 인덱스 텐서 생성 (-1은 패딩)
        self.z_indices = self._build_adjacency(code.H_z, self.max_degree)
        self.x_indices = self._build_adjacency(code.H_x, self.max_degree)
        
        # Z 안정자의 총 개수 (X 안정자 인덱스 오프셋 보정용)
        self.n_z_stabs = code.H_z.shape[0]

        # -------------------------------------------------------------------
        # 2. Embedding Layers (Patch -> Token)
        # -------------------------------------------------------------------
        # 입력: (Batch, max_degree) -> 출력: (Batch, d_model)
        self.z_embedding = nn.Linear(self.max_degree, self.d_model)
        self.x_embedding = nn.Linear(self.max_degree, self.d_model)

        # -------------------------------------------------------------------
        # 3. Positional Encodings
        # -------------------------------------------------------------------
        # 큐비트 좌표 생성 (대략적인 격자 좌표)
        coords = []
        for i in range(self.n_qubits):
            r, c = divmod(i, self.L)
            coords.append([r, c])
        coords = np.array(coords)

        # Stage 1: 2n 토큰 (각 큐비트 위치에 Z토큰, X토큰 2개가 존재하므로 좌표 중복)
        # [q0, q0, q1, q1, ...] 순서로 좌표 확장
        stage1_coords = np.repeat(coords, 2, axis=0) 
        self.pos_encoder_s1 = PositionalEncoding2D(self.d_model, dropout, stage1_coords, include_cls=False)
        
        # Stage 2: n 토큰 (큐비트당 1개)
        self.pos_encoder_s2 = PositionalEncoding2D(self.d_model, dropout, coords, include_cls=False)

        # -------------------------------------------------------------------
        # 4. Transformer Architecture
        # -------------------------------------------------------------------
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.d_model, dropout)
        ff = PositionwiseFeedForward(self.d_model, self.d_model * 4, dropout)
        
        # Stage 1: Fine-grained (Local)
        self.stage1 = Encoder(
            EncoderLayer(self.d_model, c(attn), c(ff), dropout), 
            self.n_layers
        )
        
        # Qubit Merging Layer: 2*d_model -> d_model
        # 논문 Figure 2 참조: Concatenation 후 FC Layer
        self.merge_layer = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model), # 학습 안정성을 위해 LayerNorm 추가 권장
            nn.GELU()
        )
        
        # Stage 2: Coarse-grained (Global)
        self.stage2 = Encoder(
            EncoderLayer(self.d_model, c(attn), c(ff), dropout), 
            self.n_layers
        )

        # -------------------------------------------------------------------
        # 5. Output Head
        # -------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 4) # Classes: I, X, Z, Y
        )

        # Loss Function
        if label_smoothing > 0:
            from qec.models.transformer import StructuredLabelSmoothing
            self.criterion = StructuredLabelSmoothing(smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _build_adjacency(self, H, max_degree):
        """패리티 체크 행렬 H에서 큐비트별 연결된 안정자 인덱스를 추출"""
        n_qubits = H.shape[1]
        indices = torch.full((n_qubits, max_degree), -1, dtype=torch.long)
        
        H_cpu = H.cpu() if torch.is_tensor(H) else torch.tensor(H)
        
        for q in range(n_qubits):
            # 큐비트 q와 연결된 안정자(행) 인덱스 찾기
            stabs = H_cpu[:, q].nonzero(as_tuple=False).squeeze(-1)
            deg = len(stabs)
            if deg > max_degree:
                deg = max_degree # 예외 처리 (보통 surface code는 2~4)
            indices[q, :deg] = stabs[:deg]
            
        return indices # Device 이동은 forward에서 처리하거나 init에서 처리

    def _get_patches(self, syndrome, indices, offset=0):
        """
        논문 3pg: v = 1 - 2s 변환 적용 및 패치 추출
        syndrome: (Batch, Total_Stabilizers)
        indices: (n_qubits, max_degree) - 각 큐비트가 참조할 안정자 인덱스
        offset: X 안정자의 경우 Z 안정자 개수만큼 인덱스를 밀어야 함
        """
        device = syndrome.device
        indices = indices.to(device)
        
        B = syndrome.shape[0]
        n_qubits = indices.shape[0]
        max_deg = indices.shape[1]
        
        # 1. 인덱스 오프셋 적용 (유효한 인덱스에만)
        mask = (indices != -1)
        target_indices = indices.clone()
        target_indices[mask] += offset
        
        # 2. 패딩(-1) 처리: gather를 위해 0으로 잠시 변경 (나중에 마스킹)
        safe_indices = target_indices.clone()
        safe_indices[~mask] = 0
        
        # 3. Gather를 위한 차원 확장
        # (n_qubits, max_deg) -> (B, n_qubits * max_deg)
        flat_indices = safe_indices.view(-1).unsqueeze(0).expand(B, -1)
        
        # 4. 신드롬 값 가져오기
        gathered_s = torch.gather(syndrome, 1, flat_indices) # (B, n*deg)
        
        # [cite_start]5. 값 변환: v = 1 - 2s (0 -> 1, 1 -> -1) [cite: 1]
        v = 1.0 - 2.0 * gathered_s.float()
        
        # 6. 패딩 마스크 적용 (원래 인덱스가 -1이었던 곳은 0으로 처리)
        flat_mask = mask.view(-1).unsqueeze(0).expand(B, -1).float()
        v = v * flat_mask
        
        # (B, n_qubits, max_deg) 형태로 복원
        return v.view(B, n_qubits, max_deg)

    def forward(self, x):
        """
        x: Syndrome vector (Batch, n_stabilizers)
        """
        # -------------------------------------------------------------------
        # 1. Input Processing & Embedding
        # -------------------------------------------------------------------
        # Z 패치와 X 패치 추출 (v = 1-2s 변환 포함)
        # X 인덱스는 n_z_stabs 만큼 오프셋을 더해줘야 전체 H에서의 인덱스가 됨
        z_patches = self._get_patches(x, self.z_indices, offset=0)
        x_patches = self._get_patches(x, self.x_indices, offset=self.n_z_stabs)
        
        # Embedding (Linear Projection)
        z_tokens = self.z_embedding(z_patches) # (B, n, d_model)
        x_tokens = self.x_embedding(x_patches) # (B, n, d_model)
        
        # Stage 1 입력을 위해 인터리빙 (Interleaving): [z0, x0, z1, x1, ...]
        # stack -> (B, n, 2, d) -> flatten -> (B, 2n, d)
        combined = torch.stack([z_tokens, x_tokens], dim=2)
        x_s1 = combined.view(x.shape[0], -1, self.d_model)

        # -------------------------------------------------------------------
        # 2. Stage 1 (Fine-grained)
        # -------------------------------------------------------------------
        x_s1 = self.pos_encoder_s1(x_s1) # 2n tokens
        x_s1 = self.stage1(x_s1, mask=None)

        # -------------------------------------------------------------------
        # 3. Qubit Merging
        # -------------------------------------------------------------------
        # 다시 분리: (B, 2n, d) -> (B, n, 2, d)
        x_s1_reshaped = x_s1.view(x.shape[0], self.n_qubits, 2, self.d_model)
        z_out = x_s1_reshaped[:, :, 0, :] # Z tokens
        x_out = x_s1_reshaped[:, :, 1, :] # X tokens
        
        # [cite_start]Concatenate: (B, n, 2d) [cite: 1]
        merged = torch.cat([z_out, x_out], dim=-1)
        
        # Project: (B, n, 2d) -> (B, n, d)
        x_s2_input = self.merge_layer(merged)

        # -------------------------------------------------------------------
        # 4. Stage 2 (Coarse-grained)
        # -------------------------------------------------------------------
        x_s2_input = self.pos_encoder_s2(x_s2_input) # n tokens
        x_s2 = self.stage2(x_s2_input, mask=None)

        # -------------------------------------------------------------------
        # 5. Prediction
        # -------------------------------------------------------------------
        # [cite_start]Global Mean Pooling: (B, n, d) -> (B, d) [cite: 1]
        pooled = x_s2.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)
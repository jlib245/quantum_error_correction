"""
Implementation of "Deep Quantum Error Correction" (DQEC), AAAI24
@author: Yoni Choukroun, choukroun.yoni@gmail.com
(Transformer Model Definition)
"""
from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging
from Codes import sign_to_bin, bin_to_sign
import numpy as np

###################################################
###################################################

def diff_syndrome(H,x):    
    H_bin = sign_to_bin(H) if -1 in H else H
    # x_bin = sign_to_bin(x) if -1 in x else x
    x_bin = x
    
        
    tmp = bin_to_sign(H_bin.unsqueeze(0)*x_bin.unsqueeze(-1))
    tmp = torch.prod(tmp,1)
    tmp = sign_to_bin(tmp)

    # assert torch.allclose(logical_flipped(H_bin,x_bin).cpu().detach().bool(), tmp.detach().cpu().bool())
    return tmp

def logical_flipped(L,x):
    return torch.matmul(x.float(),L.float()) % 2

###################################################
###################################################
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            if idx == len(self.layers)//2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # scores = oe.contract("x h n d, x h m d, a a n m -> x h n m", query,key,~mask)/ math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

############################################################
# --- 추가된 클래스: PositionalEncoding ---
############################################################

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

############################################################
# --- 수정된 클래스: ECC_Transformer (실제 트랜스포머 구현) ---
############################################################
    
class ECC_Transformer(nn.Module):
    def __init__(self, args, dropout=0):
        super(ECC_Transformer, self).__init__()
        self.args = args
        self.no_g = args.no_g
        code = args.code
        d_model = args.d_model
        h = args.h
        N_dec = args.N_dec
        
        c = copy.deepcopy
        
        # 트랜스포머 구성 요소 정의
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 인코더 레이어
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_dec)
        
        # 입력 임베딩: 신드롬 벡터(길이 M)를 d_model 차원으로 변환
        self.input_embedding = nn.Linear(code.pc_matrix.size(0), d_model)
        
        # 출력 분류기: 트랜스포머의 출력을 4개의 클래스(I,X,Z,Y)로 변환
        self.output_classifier = nn.Linear(d_model, 4) 
        
        self.criterion = nn.CrossEntropyLoss()
        
        # 마스크 초기화 (필요시 사용)
        self.get_mask(code, no_mask=(args.no_mask > 0))

    def forward(self, x):
        # x의 shape: (batch_size, syndrome_length)
        
        # 1. 입력 임베딩 (Batch, Syndrome_Length) -> (Batch, d_model)
        x_emb = self.input_embedding(x)
        
        # 2. 트랜스포머 입력 형태로 변경: (Batch, Sequence_Length, d_model)
        #    여기서는 신드롬 벡터 하나를 "길이가 1인 시퀀스"로 취급합니다.
        x_seq = x_emb.unsqueeze(1) 
        
        # 3. 위치 인코딩 추가
        x_pos = self.pos_encoder(x_seq)
        
        # 4. 트랜스포머 인코더 통과
        #    mask는 현재 구현에서 사용되지 않지만 (self.src_mask), 호환성을 위해 전달
        encoded_x = self.encoder(x_pos, self.src_mask) 
        
        # 5. 분류
        #    시퀀스의 첫 번째 (유일한) 토큰의 출력을 사용
        #    (Batch, 1, d_model) -> (Batch, d_model)
        seq_output = encoded_x.squeeze(1) 
        
        # (Batch, d_model) -> (Batch, 4)
        logits = self.output_classifier(seq_output)
        
        return logits

    def loss(self, pred, true_label):
        return self.criterion(pred, true_label)

    def get_mask(self, code, no_mask=False):
        # 이 트랜스포머 구현(시퀀스 길이 1)에서는 마스크가 실제 연산에 영향을 주지 않음
        # 하지만 논문 저자의 원본 코드는 마스킹 로직을 포함하고 있었으므로 유지합니다.
        if no_mask:
            self.src_mask = None
            return

        def build_mask(code):
            # 논문의 마스킹 로직 (신드롬 노드와 변수 노드 간의 연결)
            # 현재 구현에서는 신드롬만 입력으로 사용하므로, 단순한 마스크(모두 False)로 대체
            # 또는 args.no_mask=1 로 설정하여 비활성화
            syndrome_len = code.pc_matrix.size(0)
            
            # (Batch, Seq_Len, Seq_Len) 형태의 마스크
            # 현재 Seq_Len=1 이므로 (1, 1, 1) 마스크
            mask = torch.zeros(1, 1, 1, dtype=torch.bool)
            
            # 주의: 만약 신드롬 벡터 대신 (V+C) 노드 전체를 입력으로 넣는다면
            # 오리지널 마스킹 로직이 필요합니다.
            
            src_mask = mask
            return src_mask
        
        src_mask = build_mask(code)
        
        # logging.info("Using simple mask for Transformer.")
        
        self.register_buffer('src_mask', src_mask)


############################################################
############################################################

if __name__ == '__main__':
    pass
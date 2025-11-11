"""
Transformer Model for Quantum Error Correction
"""
from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging

from qec.core.codes import sign_to_bin, bin_to_sign


def diff_syndrome(H, x):
    """Calculate differential syndrome."""
    H_bin = sign_to_bin(H) if -1 in H else H
    x_bin = x

    tmp = bin_to_sign(H_bin.unsqueeze(0) * x_bin.unsqueeze(-1))
    tmp = torch.prod(tmp, 1)
    tmp = sign_to_bin(tmp)

    return tmp


def logical_flipped(L, x):
    """Check if logical operator is flipped."""
    return torch.matmul(x.float(), L.float()) % 2


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            if idx == len(self.layers) // 2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """Residual connection followed by layer norm."""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Single transformer encoder layer."""

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
    """Multi-head attention mechanism."""

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
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

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


class ECC_Transformer(nn.Module):
    """
    Transformer model for Error Correcting Codes.

    This model processes syndrome vectors through a transformer architecture
    to predict error patterns (I, X, Z, Y).
    """

    def __init__(self, args, dropout=0):
        super(ECC_Transformer, self).__init__()
        self.args = args
        self.no_g = args.no_g
        code = args.code
        d_model = args.d_model
        h = args.h
        N_dec = args.N_dec

        c = copy.deepcopy

        # Transformer components
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Encoder layers
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_dec)

        # Input embedding: syndrome vector to d_model dimensions
        self.input_embedding = nn.Linear(code.pc_matrix.size(0), d_model)

        # Output classifier: transformer output to 4 classes (I, X, Z, Y)
        self.output_classifier = nn.Linear(d_model, 4)

        self.criterion = nn.CrossEntropyLoss()

        # Initialize mask
        self.get_mask(code, no_mask=(args.no_mask > 0))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, syndrome_length)

        Returns:
            logits: Output tensor of shape (batch_size, 4)
        """
        # 1. Input embedding (Batch, Syndrome_Length) -> (Batch, d_model)
        x_emb = self.input_embedding(x)

        # 2. Reshape to sequence format: (Batch, Sequence_Length, d_model)
        x_seq = x_emb.unsqueeze(1)

        # 3. Add positional encoding
        x_pos = self.pos_encoder(x_seq)

        # 4. Pass through transformer encoder
        encoded_x = self.encoder(x_pos, self.src_mask)

        # 5. Classification
        seq_output = encoded_x.squeeze(1)
        logits = self.output_classifier(seq_output)

        return logits

    def loss(self, pred, true_label):
        """Calculate loss."""
        return self.criterion(pred, true_label)

    def get_mask(self, code, no_mask=False):
        """Initialize attention mask."""
        if no_mask:
            self.src_mask = None
            return

        def build_mask(code):
            syndrome_len = code.pc_matrix.size(0)
            mask = torch.zeros(1, 1, 1, dtype=torch.bool)
            src_mask = mask
            return src_mask

        src_mask = build_mask(code)
        self.register_buffer('src_mask', src_mask)

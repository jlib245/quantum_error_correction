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
import numpy as np

from qec.core.codes import sign_to_bin, bin_to_sign, compute_stabilizer_coordinates


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


class StructuredLabelSmoothing(nn.Module):
    """
    Structured Label Smoothing for QEC classification.

    Uses Hamming distance between error classes to distribute
    smoothing probability - closer classes get more probability mass.

    Class encoding:
        I = (z=0, x=0) - no error
        X = (z=0, x=1) - X flip only
        Z = (z=1, x=0) - Z flip only
        Y = (z=1, x=1) - both flips (correlated)
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

        # Hamming distance matrix between classes
        # I=0, X=1, Z=2, Y=3
        distance = torch.tensor([
            [0, 1, 1, 2],  # I: 1-bit to X,Z; 2-bit to Y
            [1, 0, 2, 1],  # X: 1-bit to I,Y; 2-bit to Z
            [1, 2, 0, 1],  # Z: 1-bit to I,Y; 2-bit to X
            [2, 1, 1, 0],  # Y: 1-bit to X,Z; 2-bit to I
        ], dtype=torch.float)

        # Convert distance to similarity (closer = higher)
        similarity = torch.exp(-distance)

        # Zero out diagonal and normalize each row
        for i in range(4):
            similarity[i, i] = 0
            row_sum = similarity[i].sum()
            if row_sum > 0:
                similarity[i] = similarity[i] / row_sum

        self.register_buffer('similarity', similarity)

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 4) logits
            target: (batch,) class indices
        Returns:
            loss: scalar
        """
        # Create one-hot encoding
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1.0)

        # Get structured smoothing distribution for each target
        smooth_dist = self.similarity[target]  # (batch, 4)

        # Combine: (1-ε) * one_hot + ε * structured_smooth
        smooth_label = one_hot * (1 - self.smoothing) + smooth_dist * self.smoothing

        # Cross entropy with soft labels
        log_prob = F.log_softmax(pred, dim=-1)
        loss = -(smooth_label * log_prob).sum(dim=-1).mean()

        return loss


class Encoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask, rel_pos_bias=None):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask, rel_pos_bias=rel_pos_bias)
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

    def forward(self, x, mask, rel_pos_bias=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, rel_pos_bias=rel_pos_bias))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    """Multi-head attention mechanism with optional relative position bias."""

    def __init__(self, h, d_model, dropout=0.):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, rel_pos_bias=None):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask, rel_pos_bias=rel_pos_bias)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, rel_pos_bias=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Add relative position bias if provided
        if rel_pos_bias is not None:
            # rel_pos_bias: (1, n_heads, seq_len, seq_len) or (1, 1, seq_len, seq_len)
            scores = scores + rel_pos_bias

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class RelativePositionBias(nn.Module):
    """
    Learnable relative position bias based on 2D distance.

    Converts pairwise distances between tokens into attention biases.
    """

    def __init__(self, n_heads, max_distance=10.0, n_buckets=32):
        """
        Args:
            n_heads: Number of attention heads
            max_distance: Maximum distance to consider (distances beyond this are clamped)
            n_buckets: Number of distance buckets for discretization
        """
        super(RelativePositionBias, self).__init__()
        self.n_heads = n_heads
        self.max_distance = max_distance
        self.n_buckets = n_buckets

        # Learnable bias for each bucket and head
        self.bias_embedding = nn.Embedding(n_buckets, n_heads)

    def _distance_to_bucket(self, distances):
        """Convert continuous distances to bucket indices."""
        # Normalize distances to [0, 1] range
        normalized = torch.clamp(distances / self.max_distance, 0, 1)
        # Convert to bucket indices
        buckets = (normalized * (self.n_buckets - 1)).long()
        return buckets

    def forward(self, coordinates):
        """
        Compute relative position bias matrix.

        Args:
            coordinates: (seq_len, 2) tensor of (x, y) coordinates

        Returns:
            bias: (1, n_heads, seq_len, seq_len) attention bias
        """
        seq_len = coordinates.size(0)

        # Compute pairwise Euclidean distances
        # (seq_len, 1, 2) - (1, seq_len, 2) -> (seq_len, seq_len)
        diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)
        distances = torch.norm(diff.float(), dim=-1)  # (seq_len, seq_len)

        # Convert distances to bucket indices
        bucket_ids = self._distance_to_bucket(distances)  # (seq_len, seq_len)

        # Look up biases
        bias = self.bias_embedding(bucket_ids)  # (seq_len, seq_len, n_heads)

        # Reshape to (1, n_heads, seq_len, seq_len)
        bias = bias.permute(2, 0, 1).unsqueeze(0)

        return bias


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


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for grid-structured data like surface codes.

    Uses separate sin/cos encodings for x and y coordinates, allowing
    the model to understand 2D spatial relationships.
    """

    def __init__(self, d_model, dropout, coordinates, include_cls=True):
        """
        Args:
            d_model: Model dimension
            dropout: Dropout rate
            coordinates: numpy array of shape (n_tokens, 2) with (x, y) coords
            include_cls: If True, prepend a learnable CLS position encoding
        """
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.include_cls = include_cls

        n_tokens = coordinates.shape[0]
        half_dim = d_model // 2

        # Compute 2D positional encodings
        div_term = torch.exp(torch.arange(0, half_dim, 2) *
                             -(math.log(10000.0) / half_dim))

        # X coordinate encoding (first half of d_model)
        pe_x = torch.zeros(n_tokens, half_dim)
        x_pos = torch.from_numpy(coordinates[:, 0]).float().unsqueeze(1)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term)

        # Y coordinate encoding (second half of d_model)
        pe_y = torch.zeros(n_tokens, half_dim)
        y_pos = torch.from_numpy(coordinates[:, 1]).float().unsqueeze(1)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term)

        # Concatenate x and y encodings
        pe = torch.cat([pe_x, pe_y], dim=1)  # (n_tokens, d_model)

        if include_cls:
            # Learnable CLS position encoding
            self.cls_pe = nn.Parameter(torch.randn(1, d_model) * 0.02)
            pe = pe.unsqueeze(0)  # (1, n_tokens, d_model)
        else:
            pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               If include_cls=True, seq_len should be 1 + n_tokens (CLS + tokens)
        """
        if self.include_cls:
            batch_size = x.size(0)
            # CLS token gets learnable PE, rest get 2D PE
            cls_pe = self.cls_pe.expand(batch_size, 1, -1)
            token_pe = self.pe.expand(batch_size, -1, -1)
            full_pe = torch.cat([cls_pe, token_pe], dim=1)
            x = x + full_pe.requires_grad_(False if not self.include_cls else True)
        else:
            x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class ECC_Transformer(nn.Module):
    """
    Transformer model for Error Correcting Codes.

    This model processes syndrome vectors through a transformer architecture
    to predict error patterns (I, X, Z, Y).
    """

    def __init__(self, args, dropout=0, label_smoothing=0.0):
        super(ECC_Transformer, self).__init__()
        self.args = args
        self.no_g = args.no_g
        self.label_smoothing = label_smoothing
        code = args.code
        d_model = args.d_model
        h = args.h
        N_dec = args.N_dec

        c = copy.deepcopy

        # Transformer components
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_model * 4, dropout)

        # 2D Positional encoding (graph-aware)
        # H_z and H_x check the same qubits, compute coords separately
        H_z = code.H_z.cpu().numpy()
        H_x = code.H_x.cpu().numpy()
        L = getattr(args, 'L', int(round(H_z.shape[1] ** 0.5)))

        # Compute coordinates for Z and X stabilizers separately
        coords_z = compute_stabilizer_coordinates(H_z, L)  # Z stabilizers
        coords_x = compute_stabilizer_coordinates(H_x, L)  # X stabilizers
        coordinates = np.concatenate([coords_z, coords_x], axis=0)

        self.pos_encoder = PositionalEncoding2D(d_model, dropout, coordinates, include_cls=True)

        # Store coordinates for relative position bias (add CLS position at origin)
        cls_coord = np.array([[0.0, 0.0]])  # CLS token at origin
        full_coordinates = np.concatenate([cls_coord, coordinates], axis=0)
        self.register_buffer('coordinates', torch.from_numpy(full_coordinates).float())

        # Relative position bias
        max_dist = float(L) * 1.5  # Scale with code size
        self.rel_pos_bias = RelativePositionBias(n_heads=h, max_distance=max_dist, n_buckets=32)

        # Encoder layers
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_dec)

        # Input embedding: each syndrome bit to d_model dimensions
        self.syndrome_len = code.pc_matrix.size(0)
        self.input_embedding = nn.Embedding(2, d_model)

        # Stabilizer type embedding: 0 = Z stabilizer, 1 = X stabilizer
        self.type_embedding = nn.Embedding(2, d_model)
        # Number of Z stabilizers (first block in H = block_diag(H_z, H_x))
        self.n_z_stabilizers = code.H_z.size(0)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Output classifier: CLS token output to 4 classes (I, X, Z, Y)
        self.output_classifier = nn.Linear(d_model, 4)

        # Loss function: use structured label smoothing if enabled
        if label_smoothing > 0:
            self.criterion = StructuredLabelSmoothing(smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Initialize mask (None for now, can add structure-aware mask later)
        self.src_mask = None

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, syndrome_length)

        Returns:
            logits: Output tensor of shape (batch_size, 4)
        """
        batch_size = x.size(0)
        device = x.device

        # 1. Input embedding: each syndrome bit as a token
        # (Batch, Syndrome_Length) -> (Batch, Syndrome_Length, d_model)
        x_emb = self.input_embedding(x.long())

        # 2. Add stabilizer type embedding (Z=0, X=1)
        # H = block_diag(H_z, H_x), so first n_z are Z stabilizers
        type_ids = torch.cat([
            torch.zeros(self.n_z_stabilizers, dtype=torch.long, device=device),
            torch.ones(self.syndrome_len - self.n_z_stabilizers, dtype=torch.long, device=device)
        ])
        type_emb = self.type_embedding(type_ids)  # (syndrome_len, d_model)
        x_emb = x_emb + type_emb.unsqueeze(0)  # broadcast to batch

        # 3. Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat([cls_tokens, x_emb], dim=1)  # (Batch, 1 + Syndrome_Length, d_model)

        # 4. Add positional encoding (2D graph-aware)
        x_pos = self.pos_encoder(x_seq)

        # 5. Compute relative position bias
        rel_bias = self.rel_pos_bias(self.coordinates)  # (1, n_heads, seq_len, seq_len)

        # 6. Pass through transformer encoder
        encoded_x = self.encoder(x_pos, self.src_mask, rel_pos_bias=rel_bias)

        # 7. Classification from CLS token
        cls_output = encoded_x[:, 0]  # (Batch, d_model)
        logits = self.output_classifier(cls_output)

        return logits

    def loss(self, pred, true_label):
        """Calculate loss."""
        return self.criterion(pred, true_label)


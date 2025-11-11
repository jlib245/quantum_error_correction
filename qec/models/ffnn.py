"""
Feed-Forward Neural Network Model for Quantum Error Correction
"""
import torch
import torch.nn as nn
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


class FFNN_Decoder(nn.Module):
    """
    Simple Feed-Forward Neural Network Decoder.

    Architecture based on Table 1 and Figure 5 from the paper.
    Uses a 1-hidden-layer structure with sigmoid activation.
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes=4):
        super(FFNN_Decoder, self).__init__()
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, output_nodes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_nodes)

        Returns:
            logits: Output tensor of shape (batch_size, output_nodes)
        """
        x = self.activation(self.fc1(x))
        x = self.fc2(x)  # Raw logits for CrossEntropyLoss
        return x


class ECC_FFNN(nn.Module):
    """
    Error Correcting Code Feed-Forward Neural Network.

    This is the main FFNN model used for quantum error correction,
    predicting error patterns (I, X, Z, Y) from syndrome measurements.
    """

    def __init__(self, args, dropout=0):
        super(ECC_FFNN, self).__init__()
        self.args = args
        self.no_g = args.no_g
        code = args.code

        # Simple 2-layer network
        self.fc1 = nn.Linear(code.pc_matrix.size(0), 128)
        self.fc2 = nn.Linear(128, 4)
        self.activation = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input syndrome tensor of shape (batch_size, syndrome_length)

        Returns:
            logits: Output tensor of shape (batch_size, 4)
        """
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def loss(self, pred, true_label):
        """Calculate loss."""
        return self.criterion(pred, true_label)

    def get_mask(self, code, no_mask=False):
        """
        Initialize mask (for compatibility with transformer interface).

        Note: FFNN doesn't use masks, but this method is kept for
        consistent interface with the Transformer model.
        """
        if no_mask:
            self.src_mask = None
            return

        def build_mask(code):
            mask_size = code.n + code.pc_matrix.size(0)
            mask = torch.eye(mask_size, mask_size)
            for ii in range(code.pc_matrix.size(0)):
                idx = torch.where(code.pc_matrix[ii] > 0)[0]
                for jj in idx:
                    for kk in idx:
                        if jj != kk:
                            mask[jj, kk] += 1
                            mask[kk, jj] += 1
                            mask[code.n + ii, jj] += 1
                            mask[jj, code.n + ii] += 1
            src_mask = ~(mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask

        src_mask = build_mask(code)
        mask_size = code.n + code.pc_matrix.size(0)
        a = mask_size ** 2
        logging.info(
            f'Self-Attention Sparsity Ratio={100 * torch.sum((src_mask).float()) / a:0.2f}%, '
            f'Self-Attention Complexity Ratio={100 * torch.sum((~src_mask).float()) / 2 / a:0.2f}%'
        )
        self.register_buffer('src_mask', src_mask)

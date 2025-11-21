"""
Quantum Error Correction Code Utilities
"""
import numpy as np
import torch
import os
from pathlib import Path


def sign_to_bin(x):
    """Convert sign representation to binary."""
    return 0.5 * (1 - x)


def bin_to_sign(x):
    """Convert binary to sign representation."""
    return 1 - 2 * x


def EbN0_to_std(EbN0, rate):
    """Convert Eb/N0 to standard deviation."""
    snr = EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))


def BER(x_pred, x_gt):
    """Calculate Bit Error Rate."""
    return torch.mean((x_pred != x_gt).float()).item()


def FER(x_pred, x_gt):
    """Calculate Frame Error Rate."""
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()


def _get_codes_db_path():
    """Get the absolute path to the codes_db directory."""
    # Get the path to this file's directory
    core_dir = Path(__file__).parent
    # Navigate to qec/data/codes_db
    codes_db_path = core_dir.parent / 'data' / 'codes_db'
    return codes_db_path


def Get_surface_Code(L):
    """
    Load surface code matrices.

    Args:
        L: Code distance parameter

    Returns:
        Hx, Hz, Lx, Lz: Parity check and logical operator matrices
    """
    codes_db = _get_codes_db_path()

    path_Hx = codes_db / f'Hx_surface_L{L}.txt'
    path_Lx = codes_db / f'Lx_surface_L{L}.txt'
    path_Hz = codes_db / f'Hz_surface_L{L}.txt'
    path_Lz = codes_db / f'Lz_surface_L{L}.txt'

    Hx = np.loadtxt(path_Hx)
    Lx = np.loadtxt(path_Lx)
    Hz = np.loadtxt(path_Hz)
    Lz = np.loadtxt(path_Lz)

    return Hx, Hz, Lx, Lz


def Get_toric_Code(L):
    """
    Load toric code matrices.

    Args:
        L: Code distance parameter

    Returns:
        Hx, Hz, Lx, Lz: Parity check and logical operator matrices
    """
    codes_db = _get_codes_db_path()

    path_Hx = codes_db / f'Hx_toric_L{L}.txt'
    path_Lx = codes_db / f'Lx_toric_L{L}.txt'
    path_Hz = codes_db / f'Hz_toric_L{L}.txt'
    path_Lz = codes_db / f'Lz_toric_L{L}.txt'

    Hx = np.loadtxt(path_Hx)
    Lx = np.loadtxt(path_Lx)
    Hz = np.loadtxt(path_Hz)
    Lz = np.loadtxt(path_Lz)

    return Hx, Hz, Lx, Lz


def compute_stabilizer_coordinates(H, L):
    """
    Compute 2D coordinates for stabilizers based on H matrix structure.

    For surface code, stabilizers are arranged in a 2D grid pattern.
    This function infers coordinates from the H matrix connectivity pattern.

    Args:
        H: Parity check matrix (n_stabilizers, n_qubits)
        L: Code distance (used to determine grid size)

    Returns:
        coords: numpy array of shape (n_stabilizers, 2) with (x, y) coordinates
    """
    n_stabilizers = H.shape[0]
    n_qubits = H.shape[1]

    # First, assign coordinates to data qubits (assume grid layout)
    # For surface code: n_qubits = L * L
    qubit_coords = np.zeros((n_qubits, 2))
    for i in range(n_qubits):
        qubit_coords[i] = [i % L, i // L]

    # Stabilizer coordinate = centroid of connected qubits
    stabilizer_coords = np.zeros((n_stabilizers, 2))
    for i in range(n_stabilizers):
        connected_qubits = np.where(H[i] == 1)[0]
        if len(connected_qubits) > 0:
            stabilizer_coords[i] = qubit_coords[connected_qubits].mean(axis=0)

    return stabilizer_coords


def compute_qubit_coordinates(L):
    """
    Compute 2D coordinates for data qubits.

    Args:
        L: Code distance

    Returns:
        coords: numpy array of shape (L*L, 2) with (x, y) coordinates
    """
    n_qubits = L * L
    coords = np.zeros((n_qubits, 2))
    for i in range(n_qubits):
        coords[i] = [i % L, i // L]
    return coords


# Deprecated: Legacy function for backward compatibility
def Get_toric_Code_legacy(L):
    """
    Legacy function for loading toric code (old format).
    This is deprecated and kept for backward compatibility only.
    Use Get_toric_Code instead.
    """
    codes_db = _get_codes_db_path()

    path_pc_mat = codes_db / f'H_toric_L{L}.txt'
    path_logX_mat = codes_db / f'logX_toric_L{L}.txt'

    Hx = np.loadtxt(path_pc_mat)
    logX = np.loadtxt(path_logX_mat)

    return Hx, logX

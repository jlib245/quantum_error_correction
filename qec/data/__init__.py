"""
Data module for quantum error correction codes.

This module contains:
- codes_db: Pre-computed quantum error correction code matrices
- circuit_level: Circuit-level noise model using Stim
"""

from qec.data.circuit_level import (
    create_circuit_level_surface_code,
    CircuitLevelDataset,
)

__all__ = [
    'create_circuit_level_surface_code',
    'CircuitLevelDataset',
]

"""
QEC - Quantum Error Correction Package

A Python package for quantum error correction using deep learning.
"""

__version__ = "0.1.0"

# Core functionality
from qec.core import codes

# Models
from qec.models import transformer, ffnn

__all__ = [
    "codes",
    "transformer",
    "ffnn",
]

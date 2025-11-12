"""
Decoder Evaluation Module

Provides tools for evaluating quantum error correction decoders:
- compare_decoders: Quick evaluation using mathematical simulation (training noise)
- compare_decoders_stim: Realistic evaluation using Stim-generated syndromes
"""

from .compare_decoders import evaluate_decoder, compare_all_decoders

__all__ = [
    'evaluate_decoder',
    'compare_all_decoders',
]

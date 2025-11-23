"""
Symmetry augmentation for surface code syndromes.

Surface code has D4 symmetry group (square symmetry):
- 4 rotations: 0°, 90°, 180°, 270°
- 4 reflections: horizontal, vertical, diagonal, anti-diagonal

This module provides transformations that preserve the physical meaning
of syndrome measurements while augmenting training data.
"""
import numpy as np
import torch
from functools import lru_cache


@lru_cache(maxsize=32)
def _compute_grid_indices(L: int):
    """
    Compute index mappings for L×L grid transformations.

    Returns dict with transformation matrices for each symmetry operation.
    Cached for efficiency since L is fixed during training.
    """
    n = L * L

    # Original coordinates: qubit i is at (i % L, i // L)
    coords = np.array([(i % L, i // L) for i in range(n)])

    def coord_to_idx(x, y):
        """Convert (x, y) coordinate to linear index."""
        return int(y * L + x)

    def get_permutation(transform_fn):
        """Get permutation array for a coordinate transform."""
        perm = np.zeros(n, dtype=np.int64)
        for i in range(n):
            x, y = coords[i]
            new_x, new_y = transform_fn(x, y, L)
            perm[i] = coord_to_idx(new_x, new_y)
        return perm

    # Define coordinate transformations
    transforms = {
        'identity': lambda x, y, L: (x, y),
        'rot90': lambda x, y, L: (L - 1 - y, x),           # 90° counterclockwise
        'rot180': lambda x, y, L: (L - 1 - x, L - 1 - y),  # 180°
        'rot270': lambda x, y, L: (y, L - 1 - x),          # 270° counterclockwise
        'flip_h': lambda x, y, L: (L - 1 - x, y),          # Horizontal flip
        'flip_v': lambda x, y, L: (x, L - 1 - y),          # Vertical flip
        'flip_diag': lambda x, y, L: (y, x),               # Diagonal flip (transpose)
        'flip_anti': lambda x, y, L: (L - 1 - y, L - 1 - x),  # Anti-diagonal flip
    }

    return {name: get_permutation(fn) for name, fn in transforms.items()}


def compute_syndrome_permutation(H_z, H_x, L: int, transform: str):
    """
    Compute permutation indices for syndrome transformation.

    For surface code:
    - syndrome = [s_z, s_x] where s_z = H_z @ e_x, s_x = H_x @ e_z
    - s_z has n_z stabilizers, s_x has n_x stabilizers
    - Each stabilizer's position is determined by connected qubits

    Args:
        H_z: Z stabilizer parity check matrix (n_z, n_qubits)
        H_x: X stabilizer parity check matrix (n_x, n_qubits)
        L: Code distance
        transform: Transform name ('rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', etc.)

    Returns:
        perm: Permutation array for syndrome indices
    """
    qubit_perms = _compute_grid_indices(L)
    qubit_perm = qubit_perms[transform]

    n_z = H_z.shape[0]
    n_x = H_x.shape[0]
    n_qubits = H_z.shape[1]

    # Convert to numpy if tensor
    if isinstance(H_z, torch.Tensor):
        H_z = H_z.cpu().numpy()
    if isinstance(H_x, torch.Tensor):
        H_x = H_x.cpu().numpy()

    def get_stabilizer_signature(H, stab_idx, perm):
        """Get sorted tuple of transformed qubit indices for a stabilizer."""
        connected = np.where(H[stab_idx] == 1)[0]
        transformed = tuple(sorted(perm[connected]))
        return transformed

    # Build mapping: signature -> original stabilizer index
    z_signatures = {get_stabilizer_signature(H_z, i, np.arange(n_qubits)): i
                    for i in range(n_z)}
    x_signatures = {get_stabilizer_signature(H_x, i, np.arange(n_qubits)): i
                    for i in range(n_x)}

    # Find permutation for Z stabilizers
    z_perm = np.zeros(n_z, dtype=np.int64)
    for i in range(n_z):
        sig = get_stabilizer_signature(H_z, i, qubit_perm)
        if sig in z_signatures:
            z_perm[i] = z_signatures[sig]
        else:
            # Fallback: keep original (shouldn't happen for valid transforms)
            z_perm[i] = i

    # Find permutation for X stabilizers
    x_perm = np.zeros(n_x, dtype=np.int64)
    for i in range(n_x):
        sig = get_stabilizer_signature(H_x, i, qubit_perm)
        if sig in x_signatures:
            x_perm[i] = x_signatures[sig]
        else:
            x_perm[i] = i

    # Combined permutation for syndrome = [s_z, s_x]
    full_perm = np.concatenate([z_perm, n_z + x_perm])

    return full_perm


class SyndromeAugmenter:
    """
    Efficient syndrome augmentation for training.

    Precomputes all permutation indices at initialization.
    """

    # Available transforms (D4 group)
    TRANSFORMS = ['identity', 'rot90', 'rot180', 'rot270',
                  'flip_h', 'flip_v', 'flip_diag', 'flip_anti']

    # Rotation-only transforms (C4 subgroup)
    ROTATIONS = ['identity', 'rot90', 'rot180', 'rot270']

    def __init__(self, H_z, H_x, L: int, transforms='all'):
        """
        Initialize augmenter with precomputed permutations.

        Args:
            H_z: Z stabilizer parity check matrix
            H_x: X stabilizer parity check matrix
            L: Code distance
            transforms: 'all' for D4, 'rotations' for C4, or list of transform names
        """
        self.L = L
        self.n_z = H_z.shape[0] if isinstance(H_z, np.ndarray) else H_z.shape[0]
        self.n_x = H_x.shape[0] if isinstance(H_x, np.ndarray) else H_x.shape[0]

        if transforms == 'all':
            self.transform_names = self.TRANSFORMS
        elif transforms == 'rotations':
            self.transform_names = self.ROTATIONS
        else:
            self.transform_names = transforms

        # Precompute all permutations
        self.permutations = {}
        for name in self.transform_names:
            if name != 'identity':
                perm = compute_syndrome_permutation(H_z, H_x, L, name)
                self.permutations[name] = torch.from_numpy(perm).long()

    def __call__(self, syndrome, transform=None):
        """
        Apply augmentation to syndrome.

        Args:
            syndrome: Tensor of shape (syndrome_len,) or (batch, syndrome_len)
            transform: Specific transform name, or None for random

        Returns:
            Augmented syndrome tensor
        """
        if transform is None:
            transform = np.random.choice(self.transform_names)

        if transform == 'identity':
            return syndrome

        perm = self.permutations[transform]

        if syndrome.dim() == 1:
            return syndrome[perm]
        else:
            return syndrome[:, perm]

    def random_transform(self, syndrome):
        """Apply random augmentation."""
        return self(syndrome, transform=None)

    def get_all_augmentations(self, syndrome):
        """
        Get all augmented versions of a syndrome.

        Useful for test-time augmentation (TTA).

        Returns:
            List of (transform_name, augmented_syndrome) tuples
        """
        results = []
        for name in self.transform_names:
            aug = self(syndrome, transform=name)
            results.append((name, aug))
        return results

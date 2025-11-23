"""
Symmetry augmentation for surface code syndromes.

Surface code has D4 symmetry group (square symmetry):
- 4 rotations: 0°, 90°, 180°, 270°
- 4 reflections: horizontal, vertical, diagonal, anti-diagonal

This module provides transformations that preserve the physical meaning
of syndrome measurements while augmenting training data.

IMPORTANT: Some transforms (rot90, rot270, flip_diag, flip_anti) swap Z↔X stabilizers.
This also swaps logical X ↔ logical Z, requiring label transformation:
- Class I (0) → I (0)
- Class X (1) → Z (2)
- Class Z (2) → X (1)
- Class Y (3) → Y (3)
"""
import numpy as np
import torch
from functools import lru_cache


# Transforms that swap Z↔X stabilizers (and logical X↔Z)
ZX_SWAP_TRANSFORMS = {'rot90', 'rot270', 'flip_diag', 'flip_anti'}

# Label permutation when Z↔X swap occurs: I→I, X→Z, Z→X, Y→Y
LABEL_SWAP_PERM = torch.tensor([0, 2, 1, 3])


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

    Some transforms (rot90, rot270, flip_diag, flip_anti) cause Z↔X swap:
    - Z stabilizers map to X stabilizer positions and vice versa
    - syndrome = [s_z, s_x] becomes [s_x', s_z'] after transformation

    Args:
        H_z: Z stabilizer parity check matrix (n_z, n_qubits)
        H_x: X stabilizer parity check matrix (n_x, n_qubits)
        L: Code distance
        transform: Transform name ('rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', etc.)

    Returns:
        perm: Permutation array for syndrome indices
        zx_swap: Boolean indicating if Z↔X swap occurred
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

    # Check if Z↔X swap is needed
    zx_swap = transform in ZX_SWAP_TRANSFORMS

    if not zx_swap:
        # No swap: Z→Z, X→X
        z_perm = np.zeros(n_z, dtype=np.int64)
        for i in range(n_z):
            sig = get_stabilizer_signature(H_z, i, qubit_perm)
            z_perm[i] = z_signatures.get(sig, i)

        x_perm = np.zeros(n_x, dtype=np.int64)
        for i in range(n_x):
            sig = get_stabilizer_signature(H_x, i, qubit_perm)
            x_perm[i] = x_signatures.get(sig, i)

        # syndrome = [s_z, s_x] -> [s_z[z_perm], s_x[x_perm]]
        full_perm = np.concatenate([z_perm, n_z + x_perm])
    else:
        # Swap: Z→X position, X→Z position
        # After transform, original Z stabilizers match X signatures and vice versa
        z_to_x_perm = np.zeros(n_z, dtype=np.int64)
        for i in range(n_z):
            sig = get_stabilizer_signature(H_z, i, qubit_perm)
            z_to_x_perm[i] = x_signatures.get(sig, i)

        x_to_z_perm = np.zeros(n_x, dtype=np.int64)
        for i in range(n_x):
            sig = get_stabilizer_signature(H_x, i, qubit_perm)
            x_to_z_perm[i] = z_signatures.get(sig, i)

        # syndrome = [s_z, s_x]
        # After swap: new_s_z comes from old s_x, new_s_x comes from old s_z
        # new_syndrome[i] = old_syndrome[perm[i]]
        # For positions 0..n_z-1 (new s_z): take from old s_x at x_to_z_perm
        # For positions n_z..n_z+n_x-1 (new s_x): take from old s_z at z_to_x_perm
        full_perm = np.concatenate([n_z + x_to_z_perm, z_to_x_perm])

    return full_perm, zx_swap


class SyndromeAugmenter:
    """
    Efficient syndrome augmentation for training.

    Precomputes all permutation indices at initialization.
    Handles Z↔X swap and corresponding label transformation.
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

        # Precompute all permutations and swap flags
        self.permutations = {}
        self.zx_swaps = {}
        for name in self.transform_names:
            if name != 'identity':
                perm, zx_swap = compute_syndrome_permutation(H_z, H_x, L, name)
                self.permutations[name] = torch.from_numpy(perm).long()
                self.zx_swaps[name] = zx_swap
            else:
                self.zx_swaps[name] = False

    def __call__(self, syndrome, label=None, transform=None):
        """
        Apply augmentation to syndrome and optionally label.

        Args:
            syndrome: Tensor of shape (syndrome_len,) or (batch, syndrome_len)
            label: Optional label tensor (class index 0-3)
            transform: Specific transform name, or None for random

        Returns:
            If label is None: augmented syndrome tensor
            If label is provided: (augmented_syndrome, transformed_label)
        """
        if transform is None:
            transform = np.random.choice(self.transform_names)

        # Transform syndrome
        if transform == 'identity':
            aug_syndrome = syndrome
        else:
            perm = self.permutations[transform]
            if syndrome.dim() == 1:
                aug_syndrome = syndrome[perm]
            else:
                aug_syndrome = syndrome[:, perm]

        # Transform label if provided
        if label is not None:
            if self.zx_swaps[transform]:
                # X↔Z swap: I→I, X→Z, Z→X, Y→Y
                aug_label = LABEL_SWAP_PERM[label]
            else:
                aug_label = label
            return aug_syndrome, aug_label

        return aug_syndrome

    def random_transform(self, syndrome, label=None):
        """Apply random augmentation."""
        return self(syndrome, label=label, transform=None)

    def get_all_augmentations(self, syndrome, label=None):
        """
        Get all augmented versions of a syndrome.

        Useful for test-time augmentation (TTA).

        Returns:
            List of (transform_name, augmented_syndrome, augmented_label) tuples
        """
        results = []
        for name in self.transform_names:
            result = self(syndrome, label=label, transform=name)
            if label is not None:
                aug_syn, aug_lbl = result
                results.append((name, aug_syn, aug_lbl))
            else:
                results.append((name, result))
        return results

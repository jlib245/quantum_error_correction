"""
MWPM (Minimum Weight Perfect Matching) Decoder for Surface Codes
"""
import numpy as np
import time
from tqdm import tqdm

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False


def generate_correlated_noise(n_phys, p, y_ratio):
    """Generate correlated noise with Y-error bias.

    Args:
        n_phys: Number of physical qubits
        p: Total error probability
        y_ratio: Ratio of Y errors (0 to 1)

    Returns:
        e_x, e_z: X and Z error vectors
    """
    # Calculate individual error probabilities
    p_y = p * y_ratio
    p_xz = p * (1 - y_ratio) / 2  # Split remaining between X and Z

    rand_vals = np.random.rand(n_phys)

    # Y errors (both X and Z)
    e_y = rand_vals < p_y
    # X errors only
    e_x_only = (p_y <= rand_vals) & (rand_vals < p_y + p_xz)
    # Z errors only
    e_z_only = (p_y + p_xz <= rand_vals) & (rand_vals < p_y + 2*p_xz)

    e_x = (e_y | e_x_only).astype(np.uint8)
    e_z = (e_y | e_z_only).astype(np.uint8)

    return e_x, e_z


class MWPM_Decoder:
    """MWPM decoder for surface codes using PyMatching."""

    def __init__(self, Hx, Hz, Lx, Lz):
        """Initialize MWPM decoder.

        Args:
            Hx: X stabilizer check matrix
            Hz: Z stabilizer check matrix
            Lx: X logical operator
            Lz: Z logical operator
        """
        if not HAS_PYMATCHING:
            raise ImportError("PyMatching is required for MWPM decoder. "
                            "Install with: pip install pymatching")

        self.Hx = Hx
        self.Hz = Hz
        self.Lx = Lx
        self.Lz = Lz
        self.n_phys = Hx.shape[1]

    def _create_matching(self, H, p):
        """Create PyMatching object for given check matrix and error rate."""
        # Create weights based on error probability
        weights = np.ones(H.shape[1]) * np.log((1-p)/p) if p < 0.5 else None
        return pymatching.Matching(H, weights=weights)

    def decode_single(self, syndrome_x, syndrome_z, matching_x, matching_z):
        """Decode a single syndrome.

        Args:
            syndrome_x: X syndrome (from Z errors)
            syndrome_z: Z syndrome (from X errors)
            matching_x: PyMatching object for X errors
            matching_z: PyMatching object for Z errors

        Returns:
            correction_x, correction_z: Correction vectors
        """
        # Decode Z errors using X syndrome
        correction_z = matching_z.decode(syndrome_x)
        # Decode X errors using Z syndrome
        correction_x = matching_x.decode(syndrome_z)

        return correction_x, correction_z

    def evaluate(self, p, n_shots=10000, y_ratio=0.0, verbose=False):
        """Evaluate decoder performance.

        Args:
            p: Physical error rate
            n_shots: Number of Monte Carlo shots
            y_ratio: Y-error ratio for correlated noise
            verbose: Print progress

        Returns:
            dict with 'ler', 'avg_latency', 'logical_errors', 'total_shots'
        """
        # Create matching objects
        matching_x = self._create_matching(self.Hz, p)  # For X errors
        matching_z = self._create_matching(self.Hx, p)  # For Z errors

        logical_errors = 0
        total_time = 0

        iterator = tqdm(range(n_shots)) if verbose else range(n_shots)

        for _ in iterator:
            # Generate errors
            if y_ratio > 0:
                e_x, e_z = generate_correlated_noise(self.n_phys, p, y_ratio)
            else:
                # Independent depolarizing noise
                rand_vals = np.random.rand(self.n_phys)
                e_z = (rand_vals < p/3).astype(np.uint8)
                e_x = ((p/3 <= rand_vals) & (rand_vals < 2*p/3)).astype(np.uint8)
                e_y = ((2*p/3 <= rand_vals) & (rand_vals < p)).astype(np.uint8)
                e_z = (e_z + e_y) % 2
                e_x = (e_x + e_y) % 2

            # Calculate syndromes
            syndrome_x = (self.Hx @ e_z) % 2  # X stabilizers detect Z errors
            syndrome_z = (self.Hz @ e_x) % 2  # Z stabilizers detect X errors

            # Decode
            start_time = time.perf_counter()
            correction_x, correction_z = self.decode_single(
                syndrome_x, syndrome_z, matching_x, matching_z
            )
            end_time = time.perf_counter()
            total_time += (end_time - start_time)

            # Check for logical errors
            # Total error after correction
            total_x = (e_x + correction_x) % 2
            total_z = (e_z + correction_z) % 2

            # Check if logical operators are flipped
            logical_x_flip = (self.Lz @ total_x) % 2
            logical_z_flip = (self.Lx @ total_z) % 2

            # Logical error if any logical qubit is flipped
            if np.any(logical_x_flip) or np.any(logical_z_flip):
                logical_errors += 1

        ler = logical_errors / n_shots
        avg_latency = (total_time / n_shots) * 1000  # Convert to ms

        return {
            'ler': ler,
            'avg_latency': avg_latency,
            'logical_errors': logical_errors,
            'total_shots': n_shots
        }

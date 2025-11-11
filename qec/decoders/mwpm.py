"""
Minimum Weight Perfect Matching (MWPM) Decoder

Classical baseline decoder using PyMatching library.
"""
import pymatching
import numpy as np
import time
from tqdm import tqdm


def generate_correlated_noise(n_qubits, p_total, y_ratio=0.3):
    """
    Generate correlated noise with Y errors.

    Args:
        n_qubits: Number of physical qubits
        p_total: Total error probability
        y_ratio: Ratio of Y errors (0.0~1.0)

    Returns:
        error_X, error_Z: Error vectors
    """
    p_Y = p_total * y_ratio
    p_X = p_total * (1 - y_ratio) / 2
    p_Z = p_total * (1 - y_ratio) / 2

    rand_samples = np.random.rand(n_qubits)

    error_vector_X = np.zeros(n_qubits, dtype=int)
    error_vector_Z = np.zeros(n_qubits, dtype=int)

    error_vector_X[rand_samples < p_X] = 1
    error_vector_X[(rand_samples >= p_X) & (rand_samples < p_X + p_Y)] = 1
    error_vector_Z[(rand_samples >= p_X) & (rand_samples < p_X + p_Y)] = 1
    error_vector_Z[(rand_samples >= p_X + p_Y) & (rand_samples < p_X + p_Y + p_Z)] = 1

    return error_vector_X, error_vector_Z


def generate_depolarizing_noise(n_qubits, p_error):
    """
    Generate depolarizing noise.

    Args:
        n_qubits: Number of physical qubits
        p_error: Physical error rate

    Returns:
        error_X, error_Z: Error vectors
    """
    p_channel = p_error / 3.0

    error_vector_X = (np.random.rand(n_qubits) < (p_channel * 2)).astype(int)
    error_vector_Z = (np.random.rand(n_qubits) < (p_channel * 2)).astype(int)

    return error_vector_X, error_vector_Z


class MWPM_Decoder:
    """
    Minimum Weight Perfect Matching decoder for quantum error correction.

    This is a classical baseline decoder that uses the PyMatching library
    to perform minimum weight perfect matching on the syndrome graph.
    """

    def __init__(self, Hx, Hz, Lx, Lz):
        """
        Initialize MWPM decoder.

        Args:
            Hx: X-stabilizer parity check matrix
            Hz: Z-stabilizer parity check matrix
            Lx: X-logical operator matrix
            Lz: Z-logical operator matrix
        """
        self.Hx = Hx
        self.Hz = Hz
        self.Lx = Lx
        self.Lz = Lz
        self.n_qubits = Hx.shape[1]

        # Create PyMatching decoder objects
        self.m_z = pymatching.Matching(Hx)  # Z-error decoder
        self.m_x = pymatching.Matching(Hz)  # X-error decoder

    def decode(self, syndrome_Z, syndrome_X):
        """
        Decode syndromes to estimate corrections.

        Args:
            syndrome_Z: Z-stabilizer syndrome
            syndrome_X: X-stabilizer syndrome

        Returns:
            correction_Z, correction_X: Estimated corrections
        """
        correction_Z = self.m_z.decode(syndrome_Z)
        correction_X = self.m_x.decode(syndrome_X)
        return correction_Z, correction_X

    def evaluate(self, p_error, n_shots=10000, y_ratio=0.0, verbose=True):
        """
        Evaluate decoder performance.

        Args:
            p_error: Physical error rate
            n_shots: Number of test shots
            y_ratio: Ratio of Y errors for correlated noise
            verbose: Print progress bar

        Returns:
            dict with 'ler' (logical error rate) and 'avg_latency' (ms)
        """
        logical_error_count = 0
        total_decode_time = 0

        iterator = tqdm(range(n_shots)) if verbose else range(n_shots)

        for _ in iterator:
            # Generate noise
            if y_ratio > 0:
                error_X, error_Z = generate_correlated_noise(self.n_qubits, p_error, y_ratio)
            else:
                error_X, error_Z = generate_depolarizing_noise(self.n_qubits, p_error)

            # Calculate syndromes
            syndrome_Z = self.Hx.dot(error_Z) % 2
            syndrome_X = self.Hz.dot(error_X) % 2

            # Decode
            start_time = time.perf_counter()
            correction_Z, correction_X = self.decode(syndrome_Z, syndrome_X)
            end_time = time.perf_counter()
            total_decode_time += (end_time - start_time)

            # Calculate residual error
            residual_Z = (error_Z + correction_Z) % 2
            residual_X = (error_X + correction_X) % 2

            # Check logical error
            if (self.Lx.dot(residual_Z) % 2 != 0) or (self.Lz.dot(residual_X) % 2 != 0):
                logical_error_count += 1

        ler = logical_error_count / n_shots
        avg_latency = (total_decode_time / n_shots) * 1000  # ms

        return {
            'ler': ler,
            'avg_latency': avg_latency,
            'logical_errors': logical_error_count,
            'total_shots': n_shots
        }

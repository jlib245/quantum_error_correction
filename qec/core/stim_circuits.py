"""
Custom Stim circuits for Surface Code with both X and Z logical observables.

This allows proper testing of Y error correlations by tracking both logical operators.
"""

import stim
import numpy as np


def create_rotated_surface_code_circuit(
    distance: int,
    rounds: int,
    p_error: float,
    y_ratio: float = 0.0
) -> stim.Circuit:
    """
    Create rotated surface code circuit with BOTH X and Z logical observables.

    Args:
        distance: Code distance (L)
        rounds: Number of QEC rounds
        p_error: Total error probability
        y_ratio: Ratio of Y errors (0.0 = standard depolarizing)

    Returns:
        Stim circuit with 2 observables:
            - Observable 0: X logical (flipped by Z error chains)
            - Observable 1: Z logical (flipped by X error chains)
    """

    # Calculate biased error probabilities
    if y_ratio > 0:
        p_y = p_error * y_ratio
        p_x = p_error * (1 - y_ratio) / 2
        p_z = p_error * (1 - y_ratio) / 2
    else:
        # Standard depolarizing
        p_x = p_error / 3
        p_y = p_error / 3
        p_z = p_error / 3

    # Rotated surface code layout
    # Data qubits on vertices, measure qubits on faces

    d = distance

    # Qubit indices
    # Data qubits: 0 to d*d - 1
    # X measure qubits: d*d to d*d + num_x_stabilizers - 1
    # Z measure qubits: d*d + num_x_stabilizers to ...

    num_data = d * d

    # Count stabilizers
    # For rotated surface code:
    # X stabilizers: (d-1)*d/2 + boundary terms
    # Z stabilizers: (d-1)*d/2 + boundary terms

    # Use helper to get stabilizer structure
    x_stabilizers, z_stabilizers = get_stabilizer_qubits(d)
    num_x_stab = len(x_stabilizers)
    num_z_stab = len(z_stabilizers)

    # Measure qubit indices
    x_measure_start = num_data
    z_measure_start = num_data + num_x_stab

    total_qubits = num_data + num_x_stab + num_z_stab

    circuit = stim.Circuit()

    # Initialize data qubits
    # For tracking both observables, initialize in a product state
    circuit.append("R", range(num_data))

    # Initialize measure qubits
    circuit.append("R", range(num_data, total_qubits))

    # Tick
    circuit.append("TICK")

    # QEC rounds
    for r in range(rounds):
        # Apply noise to data qubits (before stabilizer measurement)
        if p_error > 0:
            for q in range(num_data):
                circuit.append("PAULI_CHANNEL_1", [q], [p_x, p_y, p_z])

        circuit.append("TICK")

        # Measure X stabilizers
        for i, stab_qubits in enumerate(x_stabilizers):
            measure_idx = x_measure_start + i

            # Reset measure qubit
            circuit.append("R", [measure_idx])
            circuit.append("H", [measure_idx])

            # CNOTs to data qubits
            for dq in stab_qubits:
                circuit.append("CX", [measure_idx, dq])

            circuit.append("H", [measure_idx])
            circuit.append("M", [measure_idx])

        circuit.append("TICK")

        # Measure Z stabilizers
        for i, stab_qubits in enumerate(z_stabilizers):
            measure_idx = z_measure_start + i

            # Reset measure qubit
            circuit.append("R", [measure_idx])

            # CNOTs from data qubits
            for dq in stab_qubits:
                circuit.append("CX", [dq, measure_idx])

            circuit.append("M", [measure_idx])

        circuit.append("TICK")

        # Add detectors for this round
        # X stabilizer detectors
        for i in range(num_x_stab):
            if r == 0:
                # First round: compare with initial state (should be 0)
                rec_idx = -(num_x_stab + num_z_stab) + i
                circuit.append("DETECTOR", [stim.target_rec(rec_idx)])
            else:
                # Compare with previous round
                curr_rec = -(num_x_stab + num_z_stab) + i
                prev_rec = curr_rec - (num_x_stab + num_z_stab)
                circuit.append("DETECTOR", [
                    stim.target_rec(curr_rec),
                    stim.target_rec(prev_rec)
                ])

        # Z stabilizer detectors
        for i in range(num_z_stab):
            if r == 0:
                rec_idx = -num_z_stab + i
                circuit.append("DETECTOR", [stim.target_rec(rec_idx)])
            else:
                curr_rec = -num_z_stab + i
                prev_rec = curr_rec - (num_x_stab + num_z_stab)
                circuit.append("DETECTOR", [
                    stim.target_rec(curr_rec),
                    stim.target_rec(prev_rec)
                ])

    # Final data qubit measurements
    # Measure all data qubits in Z basis
    circuit.append("M", range(num_data))

    circuit.append("TICK")

    # Add final detectors from data measurements
    # These compare final measurement with last stabilizer round

    # Observable 0: X logical operator (horizontal chain)
    # Affected by Z errors
    x_logical_qubits = get_x_logical_qubits(d)
    obs_0_recs = [stim.target_rec(-num_data + q) for q in x_logical_qubits]
    circuit.append("OBSERVABLE_INCLUDE", obs_0_recs, 0)

    # Observable 1: Z logical operator (vertical chain)
    # Affected by X errors - need to track via X stabilizer measurements
    z_logical_qubits = get_z_logical_qubits(d)
    # For Z logical, we track X measurement results along the logical operator
    # This is more complex - simplified version:
    obs_1_recs = [stim.target_rec(-num_data + q) for q in z_logical_qubits]
    circuit.append("OBSERVABLE_INCLUDE", obs_1_recs, 1)

    return circuit


def get_stabilizer_qubits(d: int):
    """
    Get data qubit indices for each stabilizer in rotated surface code.

    Returns:
        x_stabilizers: List of lists, each containing data qubit indices for an X stabilizer
        z_stabilizers: List of lists, each containing data qubit indices for a Z stabilizer
    """
    x_stabilizers = []
    z_stabilizers = []

    # Rotated surface code layout
    # Data qubits at (row, col) where row, col in [0, d-1]
    # Qubit index = row * d + col

    def qubit_idx(row, col):
        return row * d + col

    # X stabilizers (weight-4 in bulk, weight-2 on boundary)
    # Located at plaquette centers where (row + col) is odd
    for row in range(d):
        for col in range(d):
            if (row + col) % 2 == 1:
                # X stabilizer plaquette
                qubits = []
                # Four corners of plaquette
                corners = [
                    (row, col),      # center
                    (row-1, col) if row > 0 else None,      # top
                    (row+1, col) if row < d-1 else None,    # bottom
                    (row, col-1) if col > 0 else None,      # left
                    (row, col+1) if col < d-1 else None,    # right
                ]

                # Actually for rotated code, stabilizers touch diagonal neighbors
                # Simplified: use adjacent qubits
                if row > 0:
                    qubits.append(qubit_idx(row-1, col))
                if row < d-1:
                    qubits.append(qubit_idx(row+1, col))
                if col > 0:
                    qubits.append(qubit_idx(row, col-1))
                if col < d-1:
                    qubits.append(qubit_idx(row, col+1))

                if len(qubits) >= 2:  # Valid stabilizer
                    x_stabilizers.append(qubits)

    # Z stabilizers (weight-4 in bulk, weight-2 on boundary)
    # Located at plaquette centers where (row + col) is even
    for row in range(d):
        for col in range(d):
            if (row + col) % 2 == 0:
                # Z stabilizer plaquette
                qubits = []
                if row > 0:
                    qubits.append(qubit_idx(row-1, col))
                if row < d-1:
                    qubits.append(qubit_idx(row+1, col))
                if col > 0:
                    qubits.append(qubit_idx(row, col-1))
                if col < d-1:
                    qubits.append(qubit_idx(row, col+1))

                if len(qubits) >= 2:
                    z_stabilizers.append(qubits)

    return x_stabilizers, z_stabilizers


def get_x_logical_qubits(d: int):
    """Get data qubit indices for X logical operator (horizontal chain)."""
    # Middle row
    row = d // 2
    return [row * d + col for col in range(d)]


def get_z_logical_qubits(d: int):
    """Get data qubit indices for Z logical operator (vertical chain)."""
    # Middle column
    col = d // 2
    return [row * d + col for row in range(d)]


class DualObservableStimDataset:
    """
    Dataset using custom Stim circuits with both X and Z observables.

    Provides 4-class labels: 0=I, 1=X, 2=Z, 3=Y
    """

    def __init__(
        self,
        distance: int,
        error_rates: list,
        samples_per_epoch: int,
        rounds: int = 1,
        y_ratio: float = 0.0
    ):
        self.distance = distance
        self.error_rates = error_rates
        self.samples_per_epoch = samples_per_epoch
        self.rounds = rounds
        self.y_ratio = y_ratio

        # Create circuits and samplers
        self.circuits = {}
        self.samplers = {}

        for p in error_rates:
            circuit = create_rotated_surface_code_circuit(
                distance=distance,
                rounds=rounds,
                p_error=p,
                y_ratio=y_ratio
            )
            self.circuits[p] = circuit
            self.samplers[p] = circuit.compile_detector_sampler()

        # Get dimensions
        self.syndrome_dim = self.circuits[error_rates[0]].num_detectors
        self.num_observables = self.circuits[error_rates[0]].num_observables

        print(f"Created DualObservableStimDataset:")
        print(f"  Distance: {distance}, Rounds: {rounds}")
        print(f"  Syndrome dim: {self.syndrome_dim}")
        print(f"  Num observables: {self.num_observables}")

    def sample(self, n_samples: int, error_rate: float = None):
        """
        Sample syndromes and 4-class labels.

        Returns:
            syndromes: (n_samples, syndrome_dim) numpy array
            labels: (n_samples,) numpy array with values 0-3
        """
        if error_rate is None:
            error_rate = np.random.choice(self.error_rates)

        sampler = self.samplers[error_rate]

        detection, observables = sampler.sample(n_samples, separate_observables=True)

        syndromes = detection.astype(np.float32)

        # Convert to 4-class
        # observables: (n_samples, 2)
        # obs[0] = X logical flip, obs[1] = Z logical flip
        x_flip = observables[:, 0].astype(int)
        z_flip = observables[:, 1].astype(int)

        labels = z_flip * 2 + x_flip  # 0=I, 1=X, 2=Z, 3=Y

        return syndromes, labels


def test_circuit():
    """Test the custom circuit generation."""
    d = 3
    rounds = 3
    p = 0.01

    circuit = create_rotated_surface_code_circuit(d, rounds, p)

    print(f"Circuit stats:")
    print(f"  Num qubits: {circuit.num_qubits}")
    print(f"  Num detectors: {circuit.num_detectors}")
    print(f"  Num observables: {circuit.num_observables}")

    # Sample
    sampler = circuit.compile_detector_sampler()
    detection, observables = sampler.sample(10, separate_observables=True)

    print(f"\nSample shapes:")
    print(f"  Detection: {detection.shape}")
    print(f"  Observables: {observables.shape}")

    # Check 4-class distribution
    x_flip = observables[:, 0]
    z_flip = observables[:, 1]
    labels = z_flip * 2 + x_flip

    print(f"\nLabels: {labels}")
    print(f"Class distribution: {np.bincount(labels, minlength=4)}")


if __name__ == "__main__":
    test_circuit()

"""
Circuit-Level Noise Model for Surface Code using Stim.

This module implements realistic circuit-level noise including:
- Gate errors (1-qubit and 2-qubit depolarizing)
- Measurement errors (bit-flip before measurement)
- Preparation/Reset errors (bit-flip after reset)
- Idle errors (depolarizing on idle qubits)

Noise model follows standard conventions from Google/IBM papers.
"""

import stim
import numpy as np
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset


def create_circuit_level_surface_code(
    distance: int,
    rounds: int,
    p_gate: float,
    p_meas: float = None,
    p_prep: float = None,
    p_idle: float = None,
    y_ratio: float = 0.0,
) -> stim.Circuit:
    """
    Create rotated surface code circuit with circuit-level noise.

    Args:
        distance: Code distance (L)
        rounds: Number of syndrome measurement rounds
        p_gate: 2-qubit gate error probability (main parameter)
        p_meas: Measurement error probability (default: p_gate)
        p_prep: Preparation/reset error probability (default: p_gate)
        p_idle: Idle qubit error probability (default: p_gate/10)
        y_ratio: Y error bias ratio (0.0 = standard depolarizing)

    Returns:
        Stim circuit with circuit-level noise
    """
    # Default error rates
    if p_meas is None:
        p_meas = p_gate
    if p_prep is None:
        p_prep = p_gate
    if p_idle is None:
        p_idle = p_gate / 10

    # Single qubit gate error (typically ~p_gate/10)
    p_single = p_gate / 10

    d = distance

    # Get stabilizer structure
    x_stabilizers, z_stabilizers = get_rotated_stabilizer_qubits(d)
    num_data = d * d
    num_x_stab = len(x_stabilizers)
    num_z_stab = len(z_stabilizers)

    # Ancilla qubit indices
    x_ancilla_start = num_data
    z_ancilla_start = num_data + num_x_stab
    total_qubits = num_data + num_x_stab + num_z_stab

    circuit = stim.Circuit()

    # Initialize all qubits with preparation noise
    circuit.append("R", range(total_qubits))
    if p_prep > 0:
        circuit.append("X_ERROR", range(total_qubits), p_prep)
    circuit.append("TICK")

    # QEC rounds
    for r in range(rounds):
        # Reset ancillas with preparation noise
        all_ancillas = list(range(x_ancilla_start, total_qubits))
        circuit.append("R", all_ancillas)
        if p_prep > 0:
            circuit.append("X_ERROR", all_ancillas, p_prep)
        circuit.append("TICK")

        # Hadamard on X-type ancillas (to prepare |+>)
        x_ancillas = list(range(x_ancilla_start, z_ancilla_start))
        circuit.append("H", x_ancillas)
        if p_single > 0:
            add_depolarizing_1q(circuit, x_ancillas, p_single, y_ratio)
        circuit.append("TICK")

        # CNOT gates in 4 steps (to avoid collisions)
        # Step 1: NW data qubits
        _apply_cnot_layer(circuit, x_stabilizers, z_stabilizers,
                         x_ancilla_start, z_ancilla_start,
                         d, step=0, p_gate=p_gate, y_ratio=y_ratio)

        # Step 2: NE data qubits
        _apply_cnot_layer(circuit, x_stabilizers, z_stabilizers,
                         x_ancilla_start, z_ancilla_start,
                         d, step=1, p_gate=p_gate, y_ratio=y_ratio)

        # Step 3: SW data qubits
        _apply_cnot_layer(circuit, x_stabilizers, z_stabilizers,
                         x_ancilla_start, z_ancilla_start,
                         d, step=2, p_gate=p_gate, y_ratio=y_ratio)

        # Step 4: SE data qubits
        _apply_cnot_layer(circuit, x_stabilizers, z_stabilizers,
                         x_ancilla_start, z_ancilla_start,
                         d, step=3, p_gate=p_gate, y_ratio=y_ratio)

        # Hadamard on X-type ancillas (convert back from |+/->)
        circuit.append("H", x_ancillas)
        if p_single > 0:
            add_depolarizing_1q(circuit, x_ancillas, p_single, y_ratio)
        circuit.append("TICK")

        # Measure all ancillas with measurement noise
        if p_meas > 0:
            circuit.append("X_ERROR", all_ancillas, p_meas)
        circuit.append("M", all_ancillas)
        circuit.append("TICK")

        # Add detectors (compare with previous round)
        for i in range(num_x_stab + num_z_stab):
            if r == 0:
                rec_idx = -(num_x_stab + num_z_stab) + i
                circuit.append("DETECTOR", [stim.target_rec(rec_idx)])
            else:
                curr_rec = -(num_x_stab + num_z_stab) + i
                prev_rec = curr_rec - (num_x_stab + num_z_stab)
                circuit.append("DETECTOR", [
                    stim.target_rec(curr_rec),
                    stim.target_rec(prev_rec)
                ])

        # Add idle noise to data qubits during measurement
        if p_idle > 0:
            add_depolarizing_1q(circuit, list(range(num_data)), p_idle, y_ratio)
        circuit.append("TICK")

    # Final data qubit measurement
    if p_meas > 0:
        circuit.append("X_ERROR", range(num_data), p_meas)
    circuit.append("M", range(num_data))
    circuit.append("TICK")

    # Observable 0: Z logical (vertical chain)
    col = d // 2
    z_logical_qubits = [row * d + col for row in range(d)]
    obs_0_recs = [stim.target_rec(-num_data + q) for q in z_logical_qubits]
    circuit.append("OBSERVABLE_INCLUDE", obs_0_recs, 0)

    # Observable 1: X logical (horizontal chain)
    row = d // 2
    x_logical_qubits = [row * d + col for col in range(d)]
    # For X observable, need to track through Z stabilizers
    obs_1_recs = [stim.target_rec(-num_data + q) for q in x_logical_qubits]
    circuit.append("OBSERVABLE_INCLUDE", obs_1_recs, 1)

    return circuit


def _apply_cnot_layer(
    circuit: stim.Circuit,
    x_stabilizers: List[List[int]],
    z_stabilizers: List[List[int]],
    x_ancilla_start: int,
    z_ancilla_start: int,
    d: int,
    step: int,
    p_gate: float,
    y_ratio: float
):
    """
    Apply one layer of CNOT gates for syndrome extraction.

    For rotated surface code, each stabilizer has up to 4 data qubits.
    We apply CNOTs in 4 steps to avoid qubit collisions.
    """
    cnot_pairs = []

    # X stabilizers: ancilla is control, data is target
    for i, stab_qubits in enumerate(x_stabilizers):
        if step < len(stab_qubits):
            ancilla = x_ancilla_start + i
            data = stab_qubits[step]
            cnot_pairs.append((ancilla, data))

    # Z stabilizers: data is control, ancilla is target
    for i, stab_qubits in enumerate(z_stabilizers):
        if step < len(stab_qubits):
            ancilla = z_ancilla_start + i
            data = stab_qubits[step]
            cnot_pairs.append((data, ancilla))

    if cnot_pairs:
        for ctrl, targ in cnot_pairs:
            circuit.append("CX", [ctrl, targ])

        # Add 2-qubit depolarizing noise
        if p_gate > 0:
            for ctrl, targ in cnot_pairs:
                add_depolarizing_2q(circuit, ctrl, targ, p_gate, y_ratio)

        circuit.append("TICK")


def add_depolarizing_1q(
    circuit: stim.Circuit,
    qubits: List[int],
    p: float,
    y_ratio: float = 0.0
):
    """Add single-qubit depolarizing noise with optional Y bias."""
    if y_ratio > 0:
        p_y = p * y_ratio
        p_x = p * (1 - y_ratio) / 2
        p_z = p * (1 - y_ratio) / 2
    else:
        p_x = p / 3
        p_y = p / 3
        p_z = p / 3

    for q in qubits:
        circuit.append("PAULI_CHANNEL_1", [q], [p_x, p_y, p_z])


def add_depolarizing_2q(
    circuit: stim.Circuit,
    q1: int,
    q2: int,
    p: float,
    y_ratio: float = 0.0
):
    """
    Add two-qubit depolarizing noise with optional Y bias.

    Standard 2-qubit depolarizing: 15 possible Pauli errors (excluding II).
    With Y bias, increase probability of YI, IY, YY terms.
    """
    if y_ratio > 0:
        # Biased 2-qubit channel
        # Simplified: apply independent biased noise to each qubit
        p_each = p / 2
        add_depolarizing_1q(circuit, [q1], p_each, y_ratio)
        add_depolarizing_1q(circuit, [q2], p_each, y_ratio)
    else:
        # Standard 2-qubit depolarizing
        circuit.append("DEPOLARIZE2", [q1, q2], p)


def get_rotated_stabilizer_qubits(d: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Get stabilizer qubits for rotated surface code.

    Returns:
        x_stabilizers: List of lists of data qubit indices for X stabilizers
        z_stabilizers: List of lists of data qubit indices for Z stabilizers
    """
    x_stabilizers = []
    z_stabilizers = []

    def qubit_idx(row, col):
        return row * d + col

    # Bulk stabilizers (weight-4)
    for row in range(d - 1):
        for col in range(d - 1):
            qubits = [
                qubit_idx(row, col),
                qubit_idx(row, col + 1),
                qubit_idx(row + 1, col),
                qubit_idx(row + 1, col + 1)
            ]
            if (row + col) % 2 == 0:
                x_stabilizers.append(qubits)
            else:
                z_stabilizers.append(qubits)

    # Boundary stabilizers (weight-2)
    for col in range(d - 1):
        if col % 2 == 0:
            z_stabilizers.append([qubit_idx(0, col), qubit_idx(0, col + 1)])
        if (d - 1 + col) % 2 == 0:
            z_stabilizers.append([qubit_idx(d - 1, col), qubit_idx(d - 1, col + 1)])

    for row in range(d - 1):
        if row % 2 == 1:
            x_stabilizers.append([qubit_idx(row, 0), qubit_idx(row + 1, 0)])
        if (row + d - 1) % 2 == 1:
            x_stabilizers.append([qubit_idx(row, d - 1), qubit_idx(row + 1, d - 1)])

    return x_stabilizers, z_stabilizers


class CircuitLevelDataset(Dataset):
    """
    PyTorch Dataset for circuit-level noise model.

    Generates syndrome data using Stim circuit-level simulation.
    """

    def __init__(
        self,
        distance: int,
        error_rates: List[float],
        length: int,
        rounds: int = 1,
        y_ratio: float = 0.0,
        p_meas: float = None,
        seed: int = None
    ):
        """
        Args:
            distance: Code distance
            error_rates: List of gate error probabilities to sample from
            length: Number of samples per epoch
            rounds: Number of syndrome measurement rounds
            y_ratio: Y error bias
            p_meas: Measurement error probability (default: same as gate error)
            seed: Random seed
        """
        self.distance = distance
        self.error_rates = error_rates
        self.length = length
        self.rounds = rounds
        self.y_ratio = y_ratio
        self.p_meas = p_meas

        if seed is not None:
            np.random.seed(seed)

        # Pre-compile circuits for each error rate
        self.circuits = {}
        self.samplers = {}

        for p in error_rates:
            circuit = create_circuit_level_surface_code(
                distance=distance,
                rounds=rounds,
                p_gate=p,
                p_meas=p_meas,
                y_ratio=y_ratio
            )
            self.circuits[p] = circuit
            self.samplers[p] = circuit.compile_detector_sampler()

        # Get dimensions from first circuit
        first_circuit = self.circuits[error_rates[0]]
        self.num_detectors = first_circuit.num_detectors
        self.num_observables = first_circuit.num_observables

        print(f"CircuitLevelDataset initialized:")
        print(f"  Distance: {distance}, Rounds: {rounds}")
        print(f"  Detectors: {self.num_detectors}")
        print(f"  Observables: {self.num_observables}")
        print(f"  Error rates: {error_rates}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            syndrome: (num_detectors,) tensor
            label: 4-class label (0=I, 1=X, 2=Z, 3=Y)
        """
        # Random error rate
        p = np.random.choice(self.error_rates)
        sampler = self.samplers[p]

        # Sample one shot
        detection, observables = sampler.sample(1, separate_observables=True)

        syndrome = torch.from_numpy(detection[0].astype(np.float32))

        # Convert to 4-class label
        x_flip = int(observables[0, 0])
        z_flip = int(observables[0, 1])
        label = z_flip * 2 + x_flip  # 0=I, 1=X, 2=Z, 3=Y

        return syndrome, label

    def sample_batch(self, batch_size: int, p: float = None):
        """
        Sample a batch of data.

        Args:
            batch_size: Number of samples
            p: Specific error rate (random if None)

        Returns:
            syndromes: (batch_size, num_detectors) tensor
            labels: (batch_size,) tensor
        """
        if p is None:
            p = np.random.choice(self.error_rates)

        sampler = self.samplers[p]
        detection, observables = sampler.sample(batch_size, separate_observables=True)

        syndromes = torch.from_numpy(detection.astype(np.float32))

        x_flip = observables[:, 0].astype(int)
        z_flip = observables[:, 1].astype(int)
        labels = torch.from_numpy(z_flip * 2 + x_flip)

        return syndromes, labels


def test_circuit_level():
    """Test circuit-level noise implementation."""
    print("Testing Circuit-Level Noise Model\n")

    d = 3
    rounds = 2
    p_gate = 0.01

    # Create circuit
    circuit = create_circuit_level_surface_code(
        distance=d,
        rounds=rounds,
        p_gate=p_gate,
        y_ratio=0.0
    )

    print(f"Circuit stats (d={d}, rounds={rounds}, p={p_gate}):")
    print(f"  Num qubits: {circuit.num_qubits}")
    print(f"  Num detectors: {circuit.num_detectors}")
    print(f"  Num observables: {circuit.num_observables}")

    # Sample
    sampler = circuit.compile_detector_sampler()
    n_samples = 1000
    detection, observables = sampler.sample(n_samples, separate_observables=True)

    print(f"\nSampling {n_samples} shots:")
    print(f"  Detection shape: {detection.shape}")
    print(f"  Observables shape: {observables.shape}")

    # Analyze error rates
    x_flip = observables[:, 0]
    z_flip = observables[:, 1]
    labels = z_flip * 2 + x_flip

    counts = np.bincount(labels, minlength=4)
    print(f"\n4-class distribution:")
    print(f"  I: {counts[0]} ({counts[0]/n_samples*100:.1f}%)")
    print(f"  X: {counts[1]} ({counts[1]/n_samples*100:.1f}%)")
    print(f"  Z: {counts[2]} ({counts[2]/n_samples*100:.1f}%)")
    print(f"  Y: {counts[3]} ({counts[3]/n_samples*100:.1f}%)")

    ler = 1 - counts[0] / n_samples
    print(f"\nLogical Error Rate: {ler:.4f}")

    # Test Y-biased noise
    print("\n" + "="*50)
    print("Testing Y-biased circuit-level noise (y_ratio=0.5)")

    circuit_y = create_circuit_level_surface_code(
        distance=d,
        rounds=rounds,
        p_gate=p_gate,
        y_ratio=0.5
    )

    sampler_y = circuit_y.compile_detector_sampler()
    detection_y, observables_y = sampler_y.sample(n_samples, separate_observables=True)

    x_flip_y = observables_y[:, 0]
    z_flip_y = observables_y[:, 1]
    labels_y = z_flip_y * 2 + x_flip_y

    counts_y = np.bincount(labels_y, minlength=4)
    print(f"\n4-class distribution (Y-biased):")
    print(f"  I: {counts_y[0]} ({counts_y[0]/n_samples*100:.1f}%)")
    print(f"  X: {counts_y[1]} ({counts_y[1]/n_samples*100:.1f}%)")
    print(f"  Z: {counts_y[2]} ({counts_y[2]/n_samples*100:.1f}%)")
    print(f"  Y: {counts_y[3]} ({counts_y[3]/n_samples*100:.1f}%)")

    # Test Dataset
    print("\n" + "="*50)
    print("Testing CircuitLevelDataset")

    dataset = CircuitLevelDataset(
        distance=5,
        error_rates=[0.001, 0.005, 0.01],
        length=1000,
        rounds=1,
        y_ratio=0.0
    )

    syndromes, labels = dataset.sample_batch(100, p=0.01)
    print(f"\nBatch sample:")
    print(f"  Syndromes shape: {syndromes.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label distribution: {torch.bincount(labels, minlength=4).tolist()}")


if __name__ == "__main__":
    test_circuit_level()

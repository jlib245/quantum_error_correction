"""
Demo: Quantum Computer Simulation

Step-by-step demonstration of how the quantum computer simulation works:
1. Stim generates realistic quantum errors
2. Syndrome measurement
3. Transformer decoding
4. Error correction
"""

import sys
import os
import numpy as np
import torch
import stim

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qec.simulation.quantum_simulator import (
    StimSurfaceCodeSimulator,
    TransformerDecoder,
    QuantumComputer
)


def demo_stim_basics():
    """Demonstrate basic Stim usage"""
    print("="*60)
    print("DEMO 1: Stim Basics")
    print("="*60)

    # Create a simple Surface Code circuit
    distance = 3
    physical_error_rate = 0.09

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=1,
        after_clifford_depolarization=physical_error_rate
    )

    print(f"\nCircuit Info:")
    print(f"  Distance: {distance}")
    print(f"  Physical qubits: {distance * distance}")
    print(f"  Detectors (syndromes): {circuit.num_detectors}")
    print(f"  Observables (logical qubits): {circuit.num_observables}")

    # Sample some syndromes
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(shots=5, separate_observables=True)

    print(f"\nSample Syndromes (first 5):")
    for i in range(5):
        syndrome = ''.join(map(str, detection_events[i].astype(int)))
        logical_error = int(observable_flips[i, 0])
        print(f"  Shot {i+1}: syndrome={syndrome}, logical_error={logical_error}")

    print()


def demo_syndrome_statistics():
    """Show syndrome statistics at different error rates"""
    print("="*60)
    print("DEMO 2: Syndrome Statistics")
    print("="*60)

    distance = 3
    shots = 1000

    error_rates = [0.05, 0.07, 0.09, 0.11, 0.13]

    print(f"\nSampling {shots} shots at each error rate:")
    print(f"{'Error Rate':>12} | {'Syndromes!=0':>14} | {'Logical Errors':>14} | {'Avg Syndrome Weight':>20}")
    print("-" * 75)

    for p in error_rates:
        simulator = StimSurfaceCodeSimulator(
            distance=distance,
            rounds=1,
            physical_error_rate=p
        )

        detection_events, observable_flips = simulator.sample_syndromes(shots)

        # Statistics
        nonzero_syndromes = (detection_events.sum(axis=1) > 0).sum()
        logical_errors = observable_flips.sum()
        avg_weight = detection_events.sum(axis=1).mean()

        print(f"{p:>12.3f} | {nonzero_syndromes:>14} | {logical_errors:>14} | {avg_weight:>20.2f}")

    print()


def demo_decoder_comparison(model_path: str = None):
    """Compare decoder predictions with actual errors"""
    print("="*60)
    print("DEMO 3: Decoder in Action")
    print("="*60)

    if model_path is None:
        print("\nNote: No model path provided. Skipping decoder demo.")
        print("Run with: python demo_quantum_computer.py --model_path <path>")
        return

    distance = 3
    physical_error_rate = 0.09
    shots = 20

    # Create simulator
    simulator = StimSurfaceCodeSimulator(
        distance=distance,
        rounds=1,
        physical_error_rate=physical_error_rate
    )

    # Sample syndromes
    detection_events, observable_flips = simulator.sample_syndromes(shots)
    syndromes = simulator._format_syndromes_for_transformer(detection_events)

    # Load decoder
    print(f"\nLoading Transformer decoder from: {model_path}")
    decoder = TransformerDecoder(model_path, device='cpu')

    # Decode
    predictions = decoder.decode_batch(syndromes)

    # Show results
    print(f"\n{'Shot':>4} | {'Syndrome':>24} | {'Actual':>8} | {'Predicted':>10} | {'Match':>6}")
    print("-" * 65)

    class_names = ['I', 'X', 'Z', 'Y']

    for i in range(min(shots, 20)):
        syndrome_str = ''.join(map(str, detection_events[i].astype(int)))
        actual = int(observable_flips[i, 0])
        predicted = int(predictions[i])
        match = "✓" if actual == predicted else "✗"

        print(f"{i+1:>4} | {syndrome_str:>24} | {class_names[actual]:>8} | {class_names[predicted]:>10} | {match:>6}")

    accuracy = (predictions[:shots] == observable_flips[:shots].flatten()).sum() / shots
    print(f"\nDecoder Accuracy: {accuracy:.2%}")
    print()


def demo_full_simulation(model_path: str = None):
    """Run full quantum computer simulation"""
    print("="*60)
    print("DEMO 4: Full Quantum Computer Simulation")
    print("="*60)

    if model_path is None:
        print("\nNote: No model path provided. Skipping full simulation.")
        return

    # Load decoder
    decoder = TransformerDecoder(model_path, device='cpu')

    # Create quantum computer
    qc = QuantumComputer(
        distance=3,
        decoder=decoder,
        physical_error_rate=0.09
    )

    # Run simulation
    result = qc.run_simulation(shots=5000, verbose=True)

    print("\nSummary:")
    print(f"  Logical Error Rate: {result.logical_error_rate:.4f}")
    print(f"  Decoder Accuracy: {result.decoder_accuracy:.2%}")
    print(f"  Execution Time: {result.execution_time:.2f}s")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Quantum Computer Demo')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained Transformer model')
    parser.add_argument('--demo', type=str, default='all',
                        choices=['all', 'stim', 'stats', 'decoder', 'full'],
                        help='Which demo to run')

    args = parser.parse_args()

    if args.demo in ['all', 'stim']:
        demo_stim_basics()

    if args.demo in ['all', 'stats']:
        demo_syndrome_statistics()

    if args.demo in ['all', 'decoder']:
        demo_decoder_comparison(args.model_path)

    if args.demo in ['all', 'full']:
        demo_full_simulation(args.model_path)

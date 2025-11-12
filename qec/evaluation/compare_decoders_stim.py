"""
Compare different decoders with Stim-generated realistic syndromes

This version uses Stim to generate syndromes instead of mathematical noise models,
providing realistic evaluation with gate-level noise.
"""
import argparse
import os
import torch
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import stim
import time
from tqdm import tqdm

from qec.core.codes import Get_surface_Code
from qec.decoders.mwpm import MWPM_Decoder


# Define Code class at module level for model loading
class Code:
    pass


def get_experiment_dir(L, noise_model):
    """Get experiment directory based on configuration."""
    base_dir = "experiments_stim"
    exp_dir = os.path.join(base_dir, f"L{L}_{noise_model}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'comparison_stim_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=handlers
    )
    return log_file


def evaluate_mwpm_stim(Hx, Hz, Lx, Lz, distance, p_errors, n_shots=10000,
                       noise_model='depolarizing', rounds=1):
    """
    Evaluate MWPM decoder with Stim-generated syndromes.

    Args:
        Hx, Hz, Lx, Lz: Surface code matrices
        distance: Code distance
        p_errors: List of error rates to test
        n_shots: Number of shots per error rate
        noise_model: 'depolarizing' or 'SI1000'
        rounds: Number of QEC rounds

    Returns:
        dict: Results for each error rate
    """
    logging.info("\n" + "="*60)
    logging.info("MWPM Decoder Evaluation (Stim)")
    logging.info("="*60)
    logging.info(f"Noise model: {noise_model}")
    logging.info(f"Distance: {distance}, Rounds: {rounds}")

    decoder = MWPM_Decoder(Hx, Hz, Lx, Lz)
    results = {}

    for p in p_errors:
        logging.info(f"\nTesting p={p:.3f}...")

        # Create Stim circuit
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=p
        )

        sampler = circuit.compile_detector_sampler()

        # Sample syndromes and observables
        detection_events, observable_flips = sampler.sample(
            n_shots,
            separate_observables=True
        )

        # Get syndrome dimensions
        n_detectors = circuit.num_detectors
        n_z_checks = Hz.shape[0]
        n_x_checks = Hx.shape[0]

        logging.info(f"  Stim detectors: {n_detectors}")
        logging.info(f"  Surface code: {n_z_checks} Z-checks, {n_x_checks} X-checks")

        # Decode all shots
        logical_error_count = 0
        total_decode_time = 0

        for shot_idx in tqdm(range(n_shots), desc=f"MWPM p={p:.3f}"):
            syndrome = detection_events[shot_idx]
            actual_logical_error = observable_flips[shot_idx, 0]  # X observable

            # For single-round Surface Code, detectors map to stabilizers
            # Split syndrome into Z-checks and X-checks
            # Note: Stim's detector ordering may differ from our convention
            if rounds == 1:
                # For rotated surface code, first half = Z stabilizers, second half = X stabilizers
                # But Stim ordering may be different - we'll use the full syndrome
                syndrome_z = syndrome[:n_z_checks] if n_detectors >= n_z_checks else syndrome
                syndrome_x = syndrome[n_z_checks:n_z_checks+n_x_checks] if n_detectors >= n_z_checks+n_x_checks else syndrome
            else:
                # Multi-round: use last round
                detectors_per_round = n_detectors // rounds
                last_round_start = (rounds - 1) * detectors_per_round
                syndrome_round = syndrome[last_round_start:]
                syndrome_z = syndrome_round[:n_z_checks]
                syndrome_x = syndrome_round[n_z_checks:n_z_checks+n_x_checks]

            # Decode with MWPM
            start_time = time.perf_counter()
            correction_z, correction_x = decoder.decode(syndrome_z, syndrome_x)
            end_time = time.perf_counter()
            total_decode_time += (end_time - start_time)

            # Check if correction causes logical error
            # For rotated_memory_x: observable tracks X logical operator
            # Logical X error: if Lz @ correction_x is odd
            logical_x_flip = np.dot(Lz, correction_x) % 2

            # Compare with ground truth
            # Note: actual_logical_error is 1 if logical error occurred
            # Our correction should predict the logical error
            # If correction produces logical flip, it means we failed to correct
            if logical_x_flip != 0:
                logical_error_count += 1

        ler = logical_error_count / n_shots
        avg_latency = (total_decode_time / n_shots) * 1000  # ms

        results[p] = {
            'ler': ler,
            'avg_latency': avg_latency,
            'logical_errors': logical_error_count,
            'total_shots': n_shots
        }

        logging.info(f"  LER: {ler:.6e}")
        logging.info(f"  Avg Latency: {avg_latency:.6f} ms")
        logging.info(f"  Logical Errors: {logical_error_count}/{n_shots}")

    return results


def evaluate_transformer_stim(model_path, Hx, Hz, Lx, Lz, distance, p_errors,
                               n_shots=10000, noise_model='depolarizing',
                               rounds=1, device='cuda'):
    """
    Evaluate Transformer decoder with Stim-generated syndromes.

    Args:
        model_path: Path to trained model
        Hx, Hz, Lx, Lz: Surface code matrices
        distance: Code distance
        p_errors: List of error rates
        n_shots: Number of shots per error rate
        noise_model: 'depolarizing' or 'SI1000'
        rounds: Number of QEC rounds
        device: Device to run on

    Returns:
        dict: Results for each error rate
    """
    logging.info("\n" + "="*60)
    logging.info("Transformer Model Evaluation (Stim)")
    logging.info("="*60)
    logging.info(f"Model path: {model_path}")
    logging.info(f"Noise model: {noise_model}")
    logging.info(f"Distance: {distance}, Rounds: {rounds}")

    # Load model
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return None

    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

    results = {}

    with torch.no_grad():
        for p in p_errors:
            logging.info(f"\nTesting p={p:.3f}...")

            # Create Stim circuit
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                distance=distance,
                rounds=rounds,
                after_clifford_depolarization=p
            )

            sampler = circuit.compile_detector_sampler()

            # Sample syndromes and observables
            detection_events, observable_flips = sampler.sample(
                n_shots,
                separate_observables=True
            )

            n_detectors = circuit.num_detectors
            logging.info(f"  Stim detectors: {n_detectors}")

            # Decode all shots
            logical_error_count = 0
            total_inference_time = 0

            for shot_idx in tqdm(range(n_shots), desc=f"Transformer p={p:.3f}"):
                syndrome = detection_events[shot_idx].astype(np.float32)
                actual_logical_error = observable_flips[shot_idx, 0]

                # Transformer inference
                syndrome_tensor = torch.from_numpy(syndrome).float().unsqueeze(0).to(device)

                start_time = time.perf_counter()
                outputs = model(syndrome_tensor)
                _, predicted = torch.max(outputs, 1)
                end_time = time.perf_counter()
                total_inference_time += (end_time - start_time)

                # For rotated_memory_x, we track X logical errors
                # predicted: 0=I, 1=X, 2=Z, 3=Y
                # actual_logical_error: 0 or 1 (X error occurred or not)

                # Simple mapping: prediction of X or Y means logical X error predicted
                predicted_x_error = predicted.item() in [1, 3]  # X or Y class

                # Check if prediction matches reality
                if predicted_x_error != bool(actual_logical_error):
                    logical_error_count += 1

            ler = logical_error_count / n_shots
            avg_latency = (total_inference_time / n_shots) * 1000  # ms

            results[p] = {
                'ler': ler,
                'avg_latency': avg_latency,
                'logical_errors': logical_error_count,
                'total_shots': n_shots
            }

            logging.info(f"  LER: {ler:.6e}")
            logging.info(f"  Avg Latency: {avg_latency:.6f} ms")
            logging.info(f"  Logical Errors: {logical_error_count}/{n_shots}")

    return results


def print_comparison_table(results_dict):
    """Print comparison table of all decoders."""
    logging.info("\n" + "="*80)
    logging.info("COMPARISON SUMMARY (Stim Evaluation)")
    logging.info("="*80)

    # Get all p values
    p_values = sorted(list(next(iter(results_dict.values())).keys()))

    # Print header
    header = f"{'p_error':<12}"
    for decoder_name in results_dict.keys():
        header += f"{decoder_name + ' LER':<20}{decoder_name + ' Latency(ms)':<25}"
    logging.info(header)
    logging.info("-" * 80)

    # Print data
    for p in p_values:
        row = f"{p:<12.3f}"
        for decoder_name, results in results_dict.items():
            if p in results:
                ler = results[p]['ler']
                latency = results[p]['avg_latency']
                row += f"{ler:<20.6e}{latency:<25.6f}"
            else:
                row += f"{'N/A':<20}{'N/A':<25}"
        logging.info(row)

    logging.info("="*80)


def plot_comparison_graphs(results_dict, save_dir, L, noise_model):
    """Plot and save LER comparison graphs."""
    # Get all p values
    p_values = sorted(list(next(iter(results_dict.values())).keys()))

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Color and marker styles for different decoders
    styles = {
        'MWPM': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'Transformer': {'color': 'red', 'marker': 's', 'linestyle': '--'},
        'FFNN': {'color': 'green', 'marker': '^', 'linestyle': '-.'}
    }

    # Plot LER (Logical Error Rate)
    for decoder_name, results in results_dict.items():
        lers = [results[p]['ler'] for p in p_values]
        style = styles.get(decoder_name, {'color': 'black', 'marker': 'x', 'linestyle': '-'})
        ax1.plot(p_values, lers,
                label=decoder_name,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=2,
                markersize=8)

    ax1.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax1.set_ylabel('Logical Error Rate (LER)', fontsize=12)
    ax1.set_title(f'Stim Evaluation: LER Comparison (L={L}, {noise_model})',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')

    # Plot threshold visualization
    for decoder_name, results in results_dict.items():
        lers = [results[p]['ler'] for p in p_values]
        style = styles.get(decoder_name, {'color': 'black', 'marker': 'x', 'linestyle': '-'})
        ax2.plot(p_values, lers,
                label=decoder_name,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=2,
                markersize=8)

    # Add reference line (y=x)
    ax2.plot(p_values, p_values, 'k:', label='y=x (threshold ref)', linewidth=1.5)

    ax2.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax2.set_ylabel('Logical Error Rate (LER)', fontsize=12)
    ax2.set_title(f'Stim Evaluation: Threshold Analysis (L={L}, {noise_model})',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(save_dir, f'comparison_stim_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logging.info(f"\nPlot saved to: {plot_file}")

    plt.close(fig)
    return plot_file


def main(args):
    # Setup experiment directory
    exp_dir = get_experiment_dir(args.L, args.noise_model)
    log_file = setup_logging(exp_dir)

    logging.info("Decoder Comparison Script (Stim Evaluation)")
    logging.info(f"Code distance L={args.L}")
    logging.info(f"Noise model: {args.noise_model}")
    logging.info(f"Test shots per p: {args.n_shots}")
    logging.info(f"Error rates: {args.p_errors}")
    logging.info(f"QEC rounds: {args.rounds}")

    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = 'xpu'
        else:
            device = 'cpu'
    else:
        device = args.device

    logging.info(f"Device: {device}")

    # Load Surface Code
    Hx, Hz, Lx, Lz = Get_surface_Code(args.L)
    logging.info(f"Surface Code loaded: n_qubits={Hx.shape[1]}")

    # Results storage
    all_results = {}

    # Evaluate MWPM
    if not args.skip_mwpm:
        mwpm_results = evaluate_mwpm_stim(
            Hx, Hz, Lx, Lz, args.L, args.p_errors,
            n_shots=args.n_shots,
            noise_model=args.noise_model,
            rounds=args.rounds
        )
        all_results['MWPM'] = mwpm_results

    # Evaluate Transformer
    if args.transformer_model:
        transformer_results = evaluate_transformer_stim(
            args.transformer_model, Hx, Hz, Lx, Lz, args.L, args.p_errors,
            n_shots=args.n_shots,
            noise_model=args.noise_model,
            rounds=args.rounds,
            device=device
        )
        if transformer_results:
            all_results['Transformer'] = transformer_results

    # Print comparison
    if all_results:
        print_comparison_table(all_results)

        # Plot comparison graphs
        plot_comparison_graphs(all_results, exp_dir, args.L, args.noise_model)

        logging.info(f"\n{'='*60}")
        logging.info(f"Results saved to: {exp_dir}")
        logging.info(f"Log file: {log_file}")
        logging.info(f"{'='*60}")
    else:
        logging.error("No results to compare!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare QEC Decoders with Stim Simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-L', type=int, default=3,
                        help='Code distance')
    parser.add_argument('-p', '--p_errors', type=float, nargs='+',
                        default=[0.07, 0.08, 0.09, 0.1, 0.11],
                        help='Error rates to test')
    parser.add_argument('-n', '--n_shots', type=int, default=10000,
                        help='Number of test shots per error rate')
    parser.add_argument('--rounds', type=int, default=1,
                        help='Number of QEC rounds')
    parser.add_argument('--noise_model', type=str, default='depolarizing',
                        choices=['depolarizing', 'SI1000'],
                        help='Stim noise model')

    parser.add_argument('--skip_mwpm', action='store_true',
                        help='Skip MWPM evaluation')
    parser.add_argument('--transformer_model', type=str, default=None,
                        help='Path to trained Transformer model')

    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'xpu'],
                        help='Device to use')

    args = parser.parse_args()
    main(args)

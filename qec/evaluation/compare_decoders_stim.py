"""
Compare Transformer and MWPM decoders with Stim-generated syndromes (4-class)

Both decoders predict X and Z logical errors for fair comparison.
"""
import argparse
import os
import torch
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pymatching
import time
from tqdm import tqdm

from qec.core.stim_circuits import create_rotated_surface_code_circuit

# Import classes needed for unpickling models trained with train_with_stim.py
from qec.training.train_with_stim import ModelArgs, Code


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
        handlers=handlers,
        force=True
    )
    return log_file


def evaluate_mwpm_stim(distance, p_errors, n_shots=10000, rounds=1, y_ratio=0.0):
    """
    Evaluate MWPM decoder with custom Stim circuit (4-class).

    Args:
        distance: Code distance
        p_errors: List of error rates to test
        n_shots: Number of shots per error rate
        rounds: Number of QEC rounds
        y_ratio: Y error ratio

    Returns:
        dict: Results for each error rate
    """
    logging.info("\n" + "="*60)
    logging.info("MWPM Decoder Evaluation (4-class)")
    logging.info("="*60)
    logging.info(f"Distance: {distance}, Rounds: {rounds}, Y_ratio: {y_ratio}")

    results = {}

    for p in p_errors:
        logging.info(f"\nTesting p={p:.3f}...")

        # Create custom circuit with both observables
        circuit = create_rotated_surface_code_circuit(
            distance=distance,
            rounds=rounds,
            p_error=p,
            y_ratio=y_ratio
        )

        # Create PyMatching decoder from detector error model
        matching = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model()
        )

        sampler = circuit.compile_detector_sampler()

        # Sample syndromes and observables
        detection_events, observable_flips = sampler.sample(
            n_shots,
            separate_observables=True
        )

        n_detectors = circuit.num_detectors
        logging.info(f"  Stim detectors: {n_detectors}")
        logging.info(f"  Num observables: {circuit.num_observables}")

        # Decode all shots
        logical_error_count = 0
        class_errors = [0, 0, 0, 0]  # I, X, Z, Y errors
        total_decode_time = 0

        for shot_idx in tqdm(range(n_shots), desc=f"MWPM p={p:.3f}"):
            syndrome = detection_events[shot_idx]

            # Actual logical errors
            actual_x = observable_flips[shot_idx, 0]
            actual_z = observable_flips[shot_idx, 1]
            actual_class = actual_z * 2 + actual_x

            # PyMatching decode
            start_time = time.perf_counter()
            predicted_observables = matching.decode(syndrome)
            end_time = time.perf_counter()
            total_decode_time += (end_time - start_time)

            # Predicted logical errors
            pred_x = predicted_observables[0]
            pred_z = predicted_observables[1]
            pred_class = pred_z * 2 + pred_x

            # Check if prediction matches
            if pred_class != actual_class:
                logical_error_count += 1
                class_errors[actual_class] += 1

        ler = logical_error_count / n_shots
        avg_latency = (total_decode_time / n_shots) * 1000  # ms

        results[p] = {
            'ler': ler,
            'avg_latency': avg_latency,
            'logical_errors': logical_error_count,
            'class_errors': class_errors,
            'total_shots': n_shots
        }

        logging.info(f"  LER: {ler:.6e}")
        logging.info(f"  Avg Latency: {avg_latency:.6f} ms")
        logging.info(f"  Logical Errors: {logical_error_count}/{n_shots}")
        logging.info(f"  Class errors (I/X/Z/Y): {class_errors}")

    return results


def evaluate_transformer_stim(model_path, distance, p_errors, n_shots=10000,
                               rounds=1, y_ratio=0.0, device='cuda'):
    """
    Evaluate Transformer decoder with custom Stim circuit (4-class).

    Args:
        model_path: Path to trained model
        distance: Code distance
        p_errors: List of error rates
        n_shots: Number of shots per error rate
        rounds: Number of QEC rounds
        y_ratio: Y error ratio
        device: Device to run on

    Returns:
        dict: Results for each error rate
    """
    logging.info("\n" + "="*60)
    logging.info("Transformer Model Evaluation (4-class)")
    logging.info("="*60)
    logging.info(f"Model path: {model_path}")
    logging.info(f"Distance: {distance}, Rounds: {rounds}, Y_ratio: {y_ratio}")

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

            # Create custom circuit with both observables
            circuit = create_rotated_surface_code_circuit(
                distance=distance,
                rounds=rounds,
                p_error=p,
                y_ratio=y_ratio
            )

            sampler = circuit.compile_detector_sampler()

            # Sample syndromes and observables
            detection_events, observable_flips = sampler.sample(
                n_shots,
                separate_observables=True
            )

            n_detectors = circuit.num_detectors
            logging.info(f"  Stim detectors: {n_detectors}")

            # Decode all shots (batch processing)
            logical_error_count = 0
            class_errors = [0, 0, 0, 0]  # I, X, Z, Y errors
            total_inference_time = 0

            batch_size = 1024
            num_batches = (n_shots + batch_size - 1) // batch_size

            for batch_idx in tqdm(range(num_batches), desc=f"Transformer p={p:.3f}"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_shots)

                # Get batch of syndromes and actual errors
                batch_syndromes = detection_events[start_idx:end_idx].astype(np.float32)
                batch_x_flips = observable_flips[start_idx:end_idx, 0]
                batch_z_flips = observable_flips[start_idx:end_idx, 1]
                batch_actual_class = batch_z_flips * 2 + batch_x_flips

                # Transformer inference (batched)
                syndrome_tensor = torch.from_numpy(batch_syndromes).float().to(device)

                start_time = time.perf_counter()
                outputs = model(syndrome_tensor)
                _, predicted = torch.max(outputs, 1)
                end_time = time.perf_counter()
                total_inference_time += (end_time - start_time)

                # Compare predictions
                predicted_np = predicted.cpu().numpy()

                for i in range(len(predicted_np)):
                    if predicted_np[i] != batch_actual_class[i]:
                        logical_error_count += 1
                        class_errors[batch_actual_class[i]] += 1

            ler = logical_error_count / n_shots
            avg_latency = (total_inference_time / n_shots) * 1000  # ms

            results[p] = {
                'ler': ler,
                'avg_latency': avg_latency,
                'logical_errors': logical_error_count,
                'class_errors': class_errors,
                'total_shots': n_shots
            }

            logging.info(f"  LER: {ler:.6e}")
            logging.info(f"  Avg Latency: {avg_latency:.6f} ms")
            logging.info(f"  Logical Errors: {logical_error_count}/{n_shots}")
            logging.info(f"  Class errors (I/X/Z/Y): {class_errors}")

    return results


def print_comparison_table(results_dict):
    """Print comparison table of all decoders."""
    logging.info("\n" + "="*80)
    logging.info("COMPARISON SUMMARY (4-class Stim Evaluation)")
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


def plot_comparison_graphs(results_dict, save_dir, L, y_ratio):
    """Plot and save LER comparison graphs."""
    # Get all p values
    p_values = sorted(list(next(iter(results_dict.values())).keys()))

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Color and marker styles for different decoders
    styles = {
        'MWPM': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'Transformer': {'color': 'red', 'marker': 's', 'linestyle': '--'},
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
    ax1.set_title(f'4-class Evaluation: LER Comparison (L={L}, y_ratio={y_ratio})',
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
    ax2.set_title(f'4-class Evaluation: Threshold Analysis (L={L}, y_ratio={y_ratio})',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(save_dir, f'comparison_4class_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logging.info(f"\nPlot saved to: {plot_file}")

    plt.close(fig)
    return plot_file


def main(args):
    # Setup experiment directory
    exp_dir = get_experiment_dir(args.L, f"y{args.y_ratio}")
    log_file = setup_logging(exp_dir)

    logging.info("Decoder Comparison Script (4-class Stim Evaluation)")
    logging.info(f"Code distance L={args.L}")
    logging.info(f"Y error ratio: {args.y_ratio}")
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

    # Results storage
    all_results = {}

    # Evaluate MWPM
    if not args.skip_mwpm:
        mwpm_results = evaluate_mwpm_stim(
            args.L, args.p_errors,
            n_shots=args.n_shots,
            rounds=args.rounds,
            y_ratio=args.y_ratio
        )
        all_results['MWPM'] = mwpm_results

    # Evaluate Transformer
    if args.transformer_model:
        transformer_results = evaluate_transformer_stim(
            args.transformer_model, args.L, args.p_errors,
            n_shots=args.n_shots,
            rounds=args.rounds,
            y_ratio=args.y_ratio,
            device=device
        )
        if transformer_results:
            all_results['Transformer'] = transformer_results

    # Print comparison
    if all_results:
        print_comparison_table(all_results)

        # Plot comparison graphs
        plot_comparison_graphs(all_results, exp_dir, args.L, args.y_ratio)

        logging.info(f"\n{'='*60}")
        logging.info(f"Results saved to: {exp_dir}")
        logging.info(f"Log file: {log_file}")
        logging.info(f"{'='*60}")
    else:
        logging.error("No results to compare!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare QEC Decoders with 4-class Stim Simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-L', type=int, default=5,
                        help='Code distance')
    parser.add_argument('-p', '--p_errors', type=float, nargs='+',
                        default=[0.01, 0.02, 0.03, 0.04, 0.05],
                        help='Error rates to test')
    parser.add_argument('-n', '--n_shots', type=int, default=10000,
                        help='Number of test shots per error rate')
    parser.add_argument('--rounds', type=int, default=1,
                        help='Number of QEC rounds')
    parser.add_argument('-y', '--y_ratio', type=float, default=0.0,
                        help='Y error ratio (0.0 = standard depolarizing)')

    parser.add_argument('--skip_mwpm', action='store_true',
                        help='Skip MWPM evaluation')
    parser.add_argument('--transformer_model', type=str, default=None,
                        help='Path to trained Transformer model')

    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'xpu'],
                        help='Device to use')

    args = parser.parse_args()
    main(args)

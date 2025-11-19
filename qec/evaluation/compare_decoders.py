"""
Compare different decoders: MWPM vs Neural Network models
"""
import argparse
import os
import torch
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from qec.core.codes import Get_surface_Code
from qec.decoders.mwpm import MWPM_Decoder


# Define Code class at module level for model loading
class Code:
    pass


def get_experiment_dir(L, y_ratio):
    """Get experiment directory based on configuration."""
    base_dir = "experiments"
    if y_ratio > 0:
        noise_type = f"L{L}_correlated_y{int(y_ratio*100):02d}"
    else:
        noise_type = f"L{L}_independent"

    exp_dir = os.path.join(base_dir, noise_type)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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


def evaluate_mwpm(Hx, Hz, Lx, Lz, p_errors, n_shots=10000, y_ratio=0.0):
    """Evaluate MWPM decoder."""
    logging.info("\n" + "="*60)
    logging.info("MWPM Decoder Evaluation")
    logging.info("="*60)

    decoder = MWPM_Decoder(Hx, Hz, Lx, Lz)
    results = {}

    for p in p_errors:
        logging.info(f"\nTesting p={p:.3f}...")
        result = decoder.evaluate(p, n_shots=n_shots, y_ratio=y_ratio, verbose=True)
        results[p] = result

        logging.info(f"  LER: {result['ler']:.6e}")
        logging.info(f"  Avg Latency: {result['avg_latency']:.6f} ms")
        logging.info(f"  Logical Errors: {result['logical_errors']}/{result['total_shots']}")

    return results


def evaluate_nn_model(model_path, model_type, Hx, Hz, Lx, Lz, p_errors,
                      n_shots=10000, y_ratio=0.0, device='cuda'):
    """Evaluate neural network decoder."""
    logging.info("\n" + "="*60)
    logging.info(f"{model_type.upper()} Model Evaluation")
    logging.info("="*60)
    logging.info(f"Model path: {model_path}")

    # Import dataset and evaluation functions
    from qec.training.train_transformer import (
        QECC_Dataset,
        create_surface_code_pure_error_lut,
        simple_decoder_C_torch
    )

    # Create pure error LUTs
    x_error_basis = create_surface_code_pure_error_lut(
        int(np.sqrt(Hx.shape[1])), 'X_only', device
    )
    z_error_basis = create_surface_code_pure_error_lut(
        int(np.sqrt(Hx.shape[1])), 'Z_only', device
    )

    # Setup code object BEFORE loading model
    code = Code()
    code.H_x = torch.from_numpy(Hx).long().to(device)
    code.H_z = torch.from_numpy(Hz).long().to(device)
    code.L_x = torch.from_numpy(Lx).long().to(device)
    code.L_z = torch.from_numpy(Lz).long().to(device)
    code.pc_matrix = torch.block_diag(code.H_z, code.H_x)
    code.logic_matrix = torch.block_diag(code.L_z, code.L_x)
    code.n = code.pc_matrix.shape[1]
    code.k = code.n - code.pc_matrix.shape[0]
    code.code_type = 'surface'

    # Create args object
    class Args:
        pass
    args = Args()
    args.y_ratio = y_ratio
    args.code = code
    args.no_g = 1

    # Load model
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return None

    model = None
    try:
        # Try loading as full model first
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        logging.info("Model loaded successfully (full model)")
    except Exception as e:
        logging.info(f"Failed to load as full model: {e}")
        logging.info("Trying to load as state_dict...")

        # Try loading as state_dict
        try:
            # Import model architecture
            if model_type.upper() == 'FFNN':
                from qec.models.ffnn import ECC_FFNN
                model = ECC_FFNN(args, dropout=0)
            elif model_type.upper() == 'TRANSFORMER':
                from qec.models.transformer import ECC_Transformer
                model = ECC_Transformer(args, dropout=0)
            else:
                logging.error(f"Unknown model type: {model_type}")
                return None

            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            logging.info("Model loaded successfully (state_dict)")
        except Exception as e2:
            logging.error(f"Failed to load model as state_dict: {e2}")
            return None

    if model is None:
        logging.error("Failed to load model")
        return None

    results = {}
    n_phys = Hx.shape[1]

    with torch.no_grad():
        for p in p_errors:
            logging.info(f"\nTesting p={p:.3f}...")

            logical_error_count = 0
            import time
            from tqdm import tqdm

            total_inference_time = 0

            for _ in tqdm(range(n_shots)):
                # Generate noise
                if y_ratio > 0:
                    from qec.decoders.mwpm import generate_correlated_noise
                    e_x_np, e_z_np = generate_correlated_noise(n_phys, p, y_ratio)
                else:
                    rand_vals = np.random.rand(n_phys)
                    e_z_np = (rand_vals < p/3)
                    e_x_np = (p/3 <= rand_vals) & (rand_vals < 2*p/3)
                    e_y_np = (2*p/3 <= rand_vals) & (rand_vals < p)
                    e_z_np, e_x_np = (e_z_np + e_y_np) % 2, (e_x_np + e_y_np) % 2

                e_z = torch.from_numpy(e_z_np).to(device, dtype=torch.uint8)
                e_x = torch.from_numpy(e_x_np).to(device, dtype=torch.uint8)
                e_full = torch.cat([e_z, e_x])

                # Calculate syndrome
                s_z_actual = (code.H_z.float() @ e_x.float()) % 2
                s_x_actual = (code.H_x.float() @ e_z.float()) % 2
                syndrome = torch.cat([s_z_actual, s_x_actual])

                # NN prediction
                start_time = time.perf_counter()
                outputs = model(syndrome.float().unsqueeze(0))
                _, predicted = torch.max(outputs.data, 1)
                end_time = time.perf_counter()
                total_inference_time += (end_time - start_time)

                # Get ground truth
                pure_error_C = simple_decoder_C_torch(
                    syndrome.type(torch.uint8),
                    x_error_basis,
                    z_error_basis,
                    code.H_z,
                    code.H_x
                )
                l_physical = pure_error_C.long() ^ e_full.long()

                l_z_physical = l_physical[:n_phys]
                l_x_physical = l_physical[n_phys:]

                l_x_flip = (code.L_z.float() @ l_x_physical.float()) % 2
                l_z_flip = (code.L_x.float() @ l_z_physical.float()) % 2

                true_class_index = (l_z_flip * 2 + l_x_flip).long()

                # Check if prediction is wrong
                if predicted.item() != true_class_index.item():
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
    logging.info("COMPARISON SUMMARY")
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
    """Plot and save LER and PER comparison graphs."""
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
    ax1.set_title(f'Logical Error Rate Comparison (L={L}, y_ratio={y_ratio})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')

    # Plot PER (Physical Error Rate vs LER - threshold visualization)
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

    # Add reference line (y=x) to visualize sub-threshold regime
    ax2.plot(p_values, p_values, 'k:', label='y=x (threshold ref)', linewidth=1.5)

    ax2.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax2.set_ylabel('Logical Error Rate (LER)', fontsize=12)
    ax2.set_title(f'Error Rate vs Threshold (L={L}, y_ratio={y_ratio})', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(save_dir, f'comparison_plot_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logging.info(f"\nPlot saved to: {plot_file}")

    # Close the figure to free memory
    plt.close(fig)

    return plot_file


def main(args):
    # Fix random seeds for reproducibility (use different seed from training)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    # Setup experiment directory
    exp_dir = get_experiment_dir(args.L, args.y_ratio)
    log_file = setup_logging(exp_dir)

    logging.info("Decoder Comparison Script")
    logging.info(f"Code distance L={args.L}")
    logging.info(f"Y-ratio: {args.y_ratio}")
    logging.info(f"Test shots per p: {args.n_shots}")
    logging.info(f"Error rates: {args.p_errors}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    # Load code
    Hx, Hz, Lx, Lz = Get_surface_Code(args.L)
    logging.info(f"Code loaded: n_qubits={Hx.shape[1]}")

    # Results storage
    all_results = {}

    # Evaluate MWPM
    if not args.skip_mwpm:
        mwpm_results = evaluate_mwpm(Hx, Hz, Lx, Lz, args.p_errors,
                                     n_shots=args.n_shots, y_ratio=args.y_ratio)
        all_results['MWPM'] = mwpm_results

    # Evaluate Transformer
    if args.transformer_model:
        transformer_results = evaluate_nn_model(
            args.transformer_model, 'Transformer',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device
        )
        if transformer_results:
            all_results['Transformer'] = transformer_results

    # Evaluate FFNN
    if args.ffnn_model:
        ffnn_results = evaluate_nn_model(
            args.ffnn_model, 'FFNN',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device
        )
        if ffnn_results:
            all_results['FFNN'] = ffnn_results

    # Print comparison
    if all_results:
        print_comparison_table(all_results)

        # Plot comparison graphs
        plot_comparison_graphs(all_results, exp_dir, args.L, args.y_ratio)
    else:
        logging.error("No results to compare!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare QEC Decoders')
    parser.add_argument('-L', type=int, default=3, help='Code distance')
    parser.add_argument('-p', '--p_errors', type=float, nargs='+',
                        default=[0.07, 0.08, 0.09, 0.1, 0.11],
                        help='Error rates to test')
    parser.add_argument('-n', '--n_shots', type=int, default=10000,
                        help='Number of test shots per error rate')
    parser.add_argument('-y', '--y_ratio', type=float, default=0.0,
                        help='Y-error ratio for correlated noise')

    parser.add_argument('--skip_mwpm', action='store_true',
                        help='Skip MWPM evaluation')
    parser.add_argument('--transformer_model', type=str, default=None,
                        help='Path to trained Transformer model')
    parser.add_argument('--ffnn_model', type=str, default=None,
                        help='Path to trained FFNN model')

    args = parser.parse_args()
    main(args)

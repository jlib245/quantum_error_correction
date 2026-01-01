"""
Compare different decoders: MWPM vs Neural Network models (Including HQMT & Jung CNN)
Supports Measurement Error simulation for 3D/Stacked models.
"""
import argparse
import os
import torch
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from qec.core.codes import Get_surface_Code
from qec.decoders.mwpm import MWPM_Decoder


def generate_noise_batch(n_phys, p, y_ratio, batch_size, device='cpu'):
    """Generate batch of correlated noise on device."""
    if y_ratio > 0:
        # Correlated noise
        p_y = p * y_ratio
        p_xz = p * (1 - y_ratio) / 2

        rand_vals = torch.rand(batch_size, n_phys, device=device)

        e_y = rand_vals < p_y
        e_x_only = (p_y <= rand_vals) & (rand_vals < p_y + p_xz)
        e_z_only = (p_y + p_xz <= rand_vals) & (rand_vals < p_y + 2*p_xz)

        e_x = (e_y | e_x_only).to(torch.uint8)
        e_z = (e_y | e_z_only).to(torch.uint8)
    else:
        # Independent depolarizing noise
        rand_vals = torch.rand(batch_size, n_phys, device=device)
        e_z = (rand_vals < p/3).to(torch.uint8)
        e_x = ((p/3 <= rand_vals) & (rand_vals < 2*p/3)).to(torch.uint8)
        e_y = ((2*p/3 <= rand_vals) & (rand_vals < p)).to(torch.uint8)
        e_z = (e_z + e_y) % 2
        e_x = (e_x + e_y) % 2

    return e_x, e_z


# Define Code class at module level for model loading
class Code:
    pass


def get_experiment_dir(L, y_ratio, p_meas):
    """Get experiment directory based on configuration."""
    base_dir = "experiments"
    if y_ratio > 0:
        noise_type = f"L{L}_correlated_y{int(y_ratio*100):02d}"
    else:
        noise_type = f"L{L}_independent"
    
    if p_meas > 0:
        noise_type += f"_pmeas{p_meas}"

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


def _reset_seed_for_idx(idx, base_seed=20_000_000):
    """Reset random seed based on index for reproducibility across decoders."""
    seed = base_seed + idx
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_mwpm(Hx, Hz, Lx, Lz, p_errors, n_shots=10000, y_ratio=0.0):
    """Evaluate MWPM decoder (2D only)."""
    logging.info("\n" + "="*60)
    logging.info("MWPM Decoder Evaluation")
    logging.info("="*60)

    decoder = MWPM_Decoder(Hx, Hz, Lx, Lz)
    results = {}

    for idx, p in enumerate(p_errors):
        _reset_seed_for_idx(idx)  # Reset seed for each p index
        logging.info(f"\nTesting p={p:.3f}...")
        result = decoder.evaluate(p, n_shots=n_shots, y_ratio=y_ratio, verbose=True)

        # Add batch latency (MWPM is sequential, so batch = avg * n_shots)
        result['batch_latency'] = result['avg_latency'] * n_shots
        result['throughput'] = n_shots / (result['batch_latency'] / 1000) if result['batch_latency'] > 0 else 0
        results[p] = result

        logging.info(f"  LER: {result['ler']:.6e}")
        logging.info(f"  Batch Latency: {result['batch_latency']:.3f} ms ({n_shots} samples)")
        logging.info(f"  Avg Latency: {result['avg_latency']:.6f} ms (per sample)")
        logging.info(f"  Throughput: {result['throughput']:,.0f} samples/sec")
        logging.info(f"  Logical Errors: {result['logical_errors']}/{result['total_shots']}")

    return results


def evaluate_nn_model(model_path, model_type, Hx, Hz, Lx, Lz, p_errors,
                      n_shots=10000, y_ratio=0.0, p_meas=0.0, device='cuda', batch_size=1024):
    """Evaluate neural network decoder."""
    logging.info("\n" + "="*60)
    logging.info(f"{model_type.upper()} Model Evaluation")
    logging.info("="*60)
    logging.info(f"Model path: {model_path}")

    # Import dataset and evaluation functions
    from qec.training.common import (
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

    # Create args object with necessary defaults
    class Args:
        pass
    args = Args()
    args.y_ratio = y_ratio
    args.code = code
    args.no_g = 1
    args.p_meas = p_meas # Add p_meas to args
    args.code_L = int(np.sqrt(Hx.shape[1]))
    
    # HQMT specific defaults
    args.d_model = 128
    args.n_heads = 4
    args.dim_feedforward = 512
    args.n_layers_stage1 = 2
    args.n_layers_stage2 = 4
    args.use_pos_enc = 1

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
        
        # Ensure model has correct attributes for inference
        if hasattr(model, 'args'):
            model.args.p_meas = p_meas # Update inference p_meas
            
    except Exception as e:
        logging.info(f"Failed to load as full model: {e}")
        logging.info("Trying to load as state_dict...")

        # Try loading as state_dict
        try:
            # Import model architecture
            if model_type.upper() == 'FFNN':
                from qec.models.ffnn import ECC_FFNN
                model = ECC_FFNN(args, dropout=0)
            elif model_type.upper() == 'CNN':
                from qec.models.cnn import ECC_CNN
                model = ECC_CNN(args, dropout=0)
            elif model_type.upper() == 'VIT':
                from qec.models.vit import ECC_ViT
                model = ECC_ViT(args, dropout=0)
            elif model_type.upper() == 'JUNG_CNN':
                from qec.models.jung_cnn import JungCNNDecoder
                model = JungCNNDecoder(args, dropout=0, label_smoothing=0)
            elif model_type.upper() == 'HQMT':
                from qec.models.hqmt import HQMT
                model = HQMT(
                    args,
                    x_error_lut=x_error_basis,
                    z_error_lut=z_error_basis,
                    dropout=0
                )
            # ... (Add other models as needed) ...
            else:
                logging.error(f"Unknown model type for state_dict loading: {model_type}")
                return None

            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            
            # Handle DataParallel wrapper
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            model.eval()
            logging.info("Model loaded successfully (state_dict)")
        except Exception as e2:
            logging.error(f"Failed to load model as state_dict: {e2}")
            return None

    model.to(device)
    results = {}
    n_phys = Hx.shape[1]

    # Determine inference mode
    use_batch = device.type in ['cuda', 'xpu']
    if not use_batch:
        batch_size = 1

    logging.info(f"Inference Mode: {'Batch' if use_batch else 'Single'}")
    if p_meas > 0:
        logging.info(f"Simulating Measurement Error (p_meas={p_meas}) with {args.code_L + 1} rounds")

    with torch.no_grad():
        for idx, p in enumerate(p_errors):
            _reset_seed_for_idx(idx)
            logging.info(f"\nTesting p={p:.3f}...")

            logical_error_count = 0
            total_inference_time = 0

            n_batches = (n_shots + batch_size - 1) // batch_size

            for batch_idx in tqdm(range(n_batches)):
                current_batch_size = min(batch_size, n_shots - batch_idx * batch_size)

                # 1. Generate Physical Errors (Data Qubits)
                e_x, e_z = generate_noise_batch(n_phys, p, y_ratio, current_batch_size, device)
                e_full = torch.cat([e_z, e_x], dim=1)

                # 2. Calculate Perfect Syndromes
                s_z_perfect = (code.H_z.float() @ e_x.float().T).T % 2
                s_x_perfect = (code.H_x.float() @ e_z.float().T).T % 2
                syndrome_perfect = torch.cat([s_z_perfect, s_x_perfect], dim=1)

                # 3. Simulate Measurement Errors if p_meas > 0
                if p_meas > 0:
                    # Time steps: usually L or L+1 for fault tolerance
                    time_steps = args.code_L + 1
                    syndromes_list = []
                    
                    # Simply repeat perfect syndrome and add measurement noise
                    # (Phenomenological noise model assumption for simple eval)
                    for t in range(time_steps):
                        meas_noise = (torch.rand_like(syndrome_perfect) < p_meas).float()
                        s_noisy = (syndrome_perfect + meas_noise) % 2
                        syndromes_list.append(s_noisy)
                    
                    # Stack: (Batch, Time, n_stab)
                    syndromes_input = torch.stack(syndromes_list, dim=1)
                else:
                    # Code Capacity: (Batch, n_stab)
                    syndromes_input = syndrome_perfect

                # 4. Neural Network Inference
                if torch.cuda.is_available(): torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                outputs = model(syndromes_input.float())
                
                # Check output format (some models return tuple)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                _, predicted = torch.max(outputs.data, 1)
                
                if torch.cuda.is_available(): torch.cuda.synchronize()
                end_time = time.perf_counter()
                total_inference_time += (end_time - start_time)

                # 5. Check Logical Errors (Ground Truth)
                # For decoding, we use the final round or perfect syndrome to check correction
                # In simple classification, we check if predicted class matches actual logical error class
                
                # Use the last noisy syndrome or perfect syndrome for standard decoder check?
                # For "Classifiers" (High-level), we compare Predicted Logical Class vs Actual Logical Error
                
                # Use perfect syndrome for ground truth lookup (since we want to correct data errors)
                for i in range(current_batch_size):
                    # Use perfect syndrome to find matching pure error
                    syndrome_i = syndrome_perfect[i].type(torch.uint8)
                    
                    pure_error_C = simple_decoder_C_torch(
                        syndrome_i,
                        x_error_basis,
                        z_error_basis,
                        code.H_z,
                        code.H_x
                    )
                    
                    # Calculate residual error
                    l_physical = pure_error_C.long() ^ e_full[i].long()

                    l_z_physical = l_physical[:n_phys]
                    l_x_physical = l_physical[n_phys:]

                    # Check logical operators
                    l_x_flip = (code.L_z.float() @ l_x_physical.float()) % 2
                    l_z_flip = (code.L_x.float() @ l_z_physical.float()) % 2

                    true_class_index = (l_z_flip * 2 + l_x_flip).long()

                    if predicted[i].item() != true_class_index.item():
                        logical_error_count += 1

            ler = logical_error_count / n_shots
            batch_latency_ms = total_inference_time * 1000
            avg_latency = batch_latency_ms / n_shots
            throughput = n_shots / total_inference_time if total_inference_time > 0 else 0

            results[p] = {
                'ler': ler,
                'avg_latency': avg_latency
            }

            logging.info(f"  LER: {ler:.6e}")
            logging.info(f"  Avg Latency: {avg_latency:.6f} ms")
            logging.info(f"  Logical Errors: {logical_error_count}/{n_shots}")

    return results


def print_comparison_table(results_dict):
    """Print comparison table."""
    logging.info("\n" + "="*80)
    logging.info("COMPARISON SUMMARY")
    logging.info("="*80)

    p_values = sorted(list(next(iter(results_dict.values())).keys()))

    header = f"{'p_error':<12}"
    for decoder_name in results_dict.keys():
        header += f"{decoder_name + ' LER':<20}"
    logging.info(header)
    logging.info("-" * 80)

    for p in p_values:
        row = f"{p:<12.3f}"
        for decoder_name, results in results_dict.items():
            if p in results:
                ler = results[p]['ler']
                row += f"{ler:<20.6e}"
            else:
                row += f"{'N/A':<20}"
        logging.info(row)
    logging.info("="*80)


def plot_comparison_graphs(results_dict, save_dir, L, y_ratio):
    """Plot graphs including HQMT and Jung CNN."""
    p_values = sorted(list(next(iter(results_dict.values())).keys()))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    styles = {
        'MWPM': {'color': 'black', 'marker': 'o', 'linestyle': '-'},
        'FFNN': {'color': 'red', 'marker': '^', 'linestyle': '-'},
        'CNN': {'color': 'green', 'marker': 'd', 'linestyle': '-'},
        'ViT': {'color': 'purple', 'marker': 'p', 'linestyle': '-'},
        
        # New Models
        'Jung_CNN': {'color': 'blue', 'marker': 'X', 'linestyle': '--', 'linewidth': 2},
        'HQMT': {'color': 'navy', 'marker': '*', 'linestyle': '-.', 'linewidth': 2},
    }

    for decoder_name, results in results_dict.items():
        lers = [results[p]['ler'] for p in p_values]
        style = styles.get(decoder_name, {'color': 'gray', 'marker': 's', 'linestyle': ':'})
        ax1.plot(p_values, lers, label=decoder_name, **style)

    ax1.set_xlabel('Physical Error Rate (p)')
    ax1.set_ylabel('Logical Error Rate (LER)')
    ax1.set_title(f'Decoder Comparison (L={L})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(save_dir, f'comparison_plot_{timestamp}.png')
    plt.savefig(plot_file, dpi=300)
    logging.info(f"Plot saved: {plot_file}")
    plt.close(fig)


def main(args):
    EVAL_SEED = 20_000_000
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)

    exp_dir = get_experiment_dir(args.L, args.y_ratio, args.p_meas)
    setup_logging(exp_dir)

    logging.info(f"Decoder Comparison (L={args.L}, p_meas={args.p_meas})")
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Device: {device}")

    Hx, Hz, Lx, Lz = Get_surface_Code(args.L)
    all_results = {}

    if not args.skip_mwpm:
        # MWPM typically ignores p_meas in this simple implementation unless updated
        mwpm_results = evaluate_mwpm(Hx, Hz, Lx, Lz, args.p_errors, args.n_shots, args.y_ratio)
        all_results['MWPM'] = mwpm_results

    # Helper to run evaluation
    def run_eval(path, name):
        if path:
            res = evaluate_nn_model(path, name, Hx, Hz, Lx, Lz, args.p_errors, 
                                  args.n_shots, args.y_ratio, args.p_meas, device, args.batch_size)
            if res: all_results[name] = res

    run_eval(args.ffnn_model, 'FFNN')
    run_eval(args.cnn_model, 'CNN')
    run_eval(args.vit_model, 'ViT')
    
    # New Models
    run_eval(args.jung_model, 'Jung_CNN')
    run_eval(args.hqmt_model, 'HQMT')

    if all_results:
        print_comparison_table(all_results)
        plot_comparison_graphs(all_results, exp_dir, args.L, args.y_ratio)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', type=int, default=3)
    parser.add_argument('-p', '--p_errors', type=float, nargs='+', default=[0.01, 0.03, 0.05])
    parser.add_argument('--p_meas', type=float, default=0.0, help='Measurement error probability')
    parser.add_argument('-n', '--n_shots', type=int, default=10000)
    parser.add_argument('-y', '--y_ratio', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--skip_mwpm', action='store_true')

    # Models
    parser.add_argument('--ffnn_model', type=str)
    parser.add_argument('--cnn_model', type=str)
    parser.add_argument('--vit_model', type=str)
    parser.add_argument('--jung_model', type=str, help='Jung CNN Model Path')
    parser.add_argument('--hqmt_model', type=str, help='HQMT Model Path')

    args = parser.parse_args()
    main(args)
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


def generate_noise_batch(n_phys, p, y_ratio, batch_size, device='cpu'):
    """Generate batch of correlated noise on device.

    Args:
        n_phys: Number of physical qubits
        p: Total error probability
        y_ratio: Ratio of Y errors (0 to 1)
        batch_size: Number of samples to generate
        device: torch device

    Returns:
        e_x, e_z: X and Z error tensors of shape (batch_size, n_phys)
    """
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


def _reset_seed_for_idx(idx, base_seed=20_000_000):
    """Reset random seed based on index for reproducibility across decoders."""
    seed = base_seed + idx
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_mwpm(Hx, Hz, Lx, Lz, p_errors, n_shots=10000, y_ratio=0.0):
    """Evaluate MWPM decoder."""
    logging.info("\n" + "="*60)
    logging.info("MWPM Decoder Evaluation")
    logging.info("="*60)

    decoder = MWPM_Decoder(Hx, Hz, Lx, Lz)
    results = {}

    for idx, p in enumerate(p_errors):
        _reset_seed_for_idx(idx)  # Reset seed for each p index
        logging.info(f"\nTesting p={p:.3f}...")
        result = decoder.evaluate(p, n_shots=n_shots, y_ratio=y_ratio, verbose=True)
        results[p] = result

        logging.info(f"  LER: {result['ler']:.6e}")
        logging.info(f"  Avg Latency: {result['avg_latency']:.6f} ms")
        logging.info(f"  Logical Errors: {result['logical_errors']}/{result['total_shots']}")

    return results


def evaluate_nn_model(model_path, model_type, Hx, Hz, Lx, Lz, p_errors,
                      n_shots=10000, y_ratio=0.0, device='cuda', batch_size=1024):
    """Evaluate neural network decoder."""
    logging.info("\n" + "="*60)
    logging.info(f"{model_type.upper()} Model Evaluation")
    logging.info("="*60)
    logging.info(f"Model path: {model_path}")

    # Import dataset and evaluation functions
    from qec.training.common import (
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

        # For ViT models, recompute coord maps to match current code
        if model_type.upper() in ['VIT', 'VIT_LARGE', 'VIT_DUALGRID']:
            from qec.models.vit import compute_stabilizer_positions_from_H
            L = int(np.sqrt(Hx.shape[1]))
            model.L = L
            model.grid_size = L
            model.n_z = Hz.shape[0]
            model.n_x = Hx.shape[0]
            model.z_coord_map = compute_stabilizer_positions_from_H(Hz, L)
            model.x_coord_map = compute_stabilizer_positions_from_H(Hx, L)
            # For DualGrid, also update H matrices
            if model_type.upper() == 'VIT_DUALGRID':
                device = next(model.parameters()).device
                model.H_z = torch.from_numpy(Hz).float().to(device)
                model.H_x = torch.from_numpy(Hx).float().to(device)
                model.n_qubits = Hx.shape[1]
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
            elif model_type.upper() == 'CNN':
                from qec.models.cnn import ECC_CNN
                model = ECC_CNN(args, dropout=0)
            elif model_type.upper() == 'CNN_LARGE':
                from qec.models.cnn import ECC_CNN_Large
                model = ECC_CNN_Large(args, dropout=0)
            elif model_type.upper() == 'VIT':
                from qec.models.vit import ECC_ViT
                args.code_L = int(np.sqrt(Hx.shape[1]))
                model = ECC_ViT(args, dropout=0)
            elif model_type.upper() == 'VIT_LARGE':
                from qec.models.vit import ECC_ViT_Large
                args.code_L = int(np.sqrt(Hx.shape[1]))
                model = ECC_ViT_Large(args, dropout=0)
            elif model_type.upper() == 'QUBIT_CENTRIC':
                from qec.models.qubit_centric import ECC_QubitCentric
                args.code_L = int(np.sqrt(Hx.shape[1]))
                model = ECC_QubitCentric(args, dropout=0)
            elif model_type.upper() == 'LUT_RESIDUAL':
                from qec.models.qubit_centric import ECC_LUT_Residual
                args.code_L = int(np.sqrt(Hx.shape[1]))
                model = ECC_LUT_Residual(args, x_error_basis, z_error_basis, dropout=0)
            elif model_type.upper() == 'LUT_CONCAT':
                from qec.models.qubit_centric import ECC_LUT_Concat
                args.code_L = int(np.sqrt(Hx.shape[1]))
                model = ECC_LUT_Concat(args, x_error_basis, z_error_basis, dropout=0)
            elif model_type.upper() == 'DIAMOND':
                from qec.models.diamond_cnn import ECC_DiamondCNN
                args.code_L = int(np.sqrt(Hx.shape[1]))
                model = ECC_DiamondCNN(
                    args,
                    x_error_lut=x_error_basis,
                    z_error_lut=z_error_basis,
                    dropout=0
                )
            elif model_type.upper() == 'DIAMOND_DEEP':
                from qec.models.diamond_cnn import ECC_DiamondCNN_Deep
                args.code_L = int(np.sqrt(Hx.shape[1]))
                model = ECC_DiamondCNN_Deep(
                    args,
                    x_error_lut=x_error_basis,
                    z_error_lut=z_error_basis,
                    dropout=0
                )
            elif model_type.upper() == 'VIT_QUBIT_CENTRIC':
                from qec.models.vit import ECC_ViT_QubitCentric
                args.code_L = int(np.sqrt(Hx.shape[1]))
                model = ECC_ViT_QubitCentric(args, dropout=0)
            elif model_type.upper() == 'VIT_LUT_CONCAT':
                from qec.models.vit import ECC_ViT_LUT_Concat
                args.code_L = int(np.sqrt(Hx.shape[1]))
                model = ECC_ViT_LUT_Concat(args, x_error_basis, z_error_basis, dropout=0)
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

    # Determine if we should use batch inference (GPU/XPU only)
    use_batch = device.type in ['cuda', 'xpu']
    if not use_batch:
        batch_size = 1

    if use_batch:
        logging.info(f"Using batch inference (batch_size={batch_size})")
    else:
        logging.info("Using single-sample inference (CPU mode)")

    import time
    from tqdm import tqdm

    with torch.no_grad():
        for idx, p in enumerate(p_errors):
            _reset_seed_for_idx(idx)  # Reset seed for each p index
            logging.info(f"\nTesting p={p:.3f}...")

            logical_error_count = 0
            total_inference_time = 0

            if use_batch:
                # Batch inference for GPU/XPU
                n_batches = (n_shots + batch_size - 1) // batch_size

                for batch_idx in tqdm(range(n_batches)):
                    current_batch_size = min(batch_size, n_shots - batch_idx * batch_size)

                    # Generate batch noise on device
                    e_x, e_z = generate_noise_batch(n_phys, p, y_ratio, current_batch_size, device)

                    # Calculate syndromes (batch)
                    # s_z = H_z @ e_x^T, s_x = H_x @ e_z^T
                    s_z = (code.H_z.float() @ e_x.float().T).T % 2  # (batch, n_z_stab)
                    s_x = (code.H_x.float() @ e_z.float().T).T % 2  # (batch, n_x_stab)
                    syndromes = torch.cat([s_z, s_x], dim=1)  # (batch, n_stab)

                    # Batch NN prediction
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                        torch.xpu.synchronize()

                    start_time = time.perf_counter()
                    outputs = model(syndromes.float())
                    _, predicted = torch.max(outputs.data, 1)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                        torch.xpu.synchronize()

                    end_time = time.perf_counter()
                    total_inference_time += (end_time - start_time)

                    # Compute ground truth (batch)
                    e_full = torch.cat([e_z, e_x], dim=1)  # (batch, 2*n_phys)

                    # Get pure error for each syndrome in batch
                    for i in range(current_batch_size):
                        syndrome_i = syndromes[i].type(torch.uint8)
                        pure_error_C = simple_decoder_C_torch(
                            syndrome_i,
                            x_error_basis,
                            z_error_basis,
                            code.H_z,
                            code.H_x
                        )
                        l_physical = pure_error_C.long() ^ e_full[i].long()

                        l_z_physical = l_physical[:n_phys]
                        l_x_physical = l_physical[n_phys:]

                        l_x_flip = (code.L_z.float() @ l_x_physical.float()) % 2
                        l_z_flip = (code.L_x.float() @ l_z_physical.float()) % 2

                        true_class_index = (l_z_flip * 2 + l_x_flip).long()

                        if predicted[i].item() != true_class_index.item():
                            logical_error_count += 1
            else:
                # Single-sample inference for CPU
                from qec.decoders.mwpm import generate_correlated_noise

                for _ in tqdm(range(n_shots)):
                    # Generate noise
                    if y_ratio > 0:
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

                    if predicted.item() != true_class_index.item():
                        logical_error_count += 1

            ler = logical_error_count / n_shots
            avg_latency = (total_inference_time / n_shots) * 1000  # ms (per sample, amortized for batch)

            results[p] = {
                'ler': ler,
                'avg_latency': avg_latency,
                'logical_errors': logical_error_count,
                'total_shots': n_shots,
                'batch_size': batch_size if use_batch else 1
            }

            logging.info(f"  LER: {ler:.6e}")
            logging.info(f"  Avg Latency: {avg_latency:.6f} ms (per sample, {'batch amortized' if use_batch else 'single'})")
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
    # Regular models: solid lines with circle-like markers
    # Code Capacity models: dashed lines with different markers
    styles = {
        # Classical decoder
        'MWPM': {'color': '#000000', 'marker': 'o', 'linestyle': '-'},  # black

        # Regular models (solid lines) - warm colors
        'Transformer': {'color': '#1f77b4', 'marker': 's', 'linestyle': '-'},  # blue
        'FFNN': {'color': '#d62728', 'marker': '^', 'linestyle': '-'},  # red
        'CNN': {'color': '#2ca02c', 'marker': 'd', 'linestyle': '-'},  # green
        'CNN_Large': {'color': '#ff7f0e', 'marker': 'v', 'linestyle': '-'},  # orange
        'ViT': {'color': '#9467bd', 'marker': 'p', 'linestyle': '-'},  # purple
        'ViT_Large': {'color': '#8c564b', 'marker': 'h', 'linestyle': '-'},  # brown

        # Code Capacity models (dashed lines) - cool/distinct colors
        'Diamond': {'color': '#17becf', 'marker': 'D', 'linestyle': '--'},  # cyan
        'Diamond_Deep': {'color': '#bcbd22', 'marker': 'H', 'linestyle': '--'},  # olive
        'LUT_Concat': {'color': '#e377c2', 'marker': 'P', 'linestyle': '--'},  # pink
        'LUT_Residual': {'color': '#7f7f7f', 'marker': 'X', 'linestyle': '--'},  # gray
        'QubitCentric': {'color': '#aec7e8', 'marker': '<', 'linestyle': '--'},  # light blue
        'ViT_QubitCentric': {'color': '#ffbb78', 'marker': '>', 'linestyle': '--'},  # light orange
        'ViT_LUT_Concat': {'color': '#98df8a', 'marker': '*', 'linestyle': '--'},  # light green
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
    # Fix random seeds for reproducibility
    # Use large offset to ensure no overlap with training (seed 42~) or test (seed 10M~)
    EVAL_SEED = 20_000_000
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EVAL_SEED)

    # Setup experiment directory
    exp_dir = get_experiment_dir(args.L, args.y_ratio)
    log_file = setup_logging(exp_dir)

    logging.info("Decoder Comparison Script")
    logging.info(f"Code distance L={args.L}")
    logging.info(f"Y-ratio: {args.y_ratio}")
    logging.info(f"Test shots per p: {args.n_shots}")
    logging.info(f"Error rates: {args.p_errors}")

    # Setup device based on args
    def check_xpu_available():
        """Check if XPU is available (Intel GPU via PyTorch 2.9+ native support)"""
        try:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return True
        except Exception:
            pass
        return False

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif check_xpu_available():
            device = torch.device('xpu')
        else:
            device = torch.device('cpu')
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logging.warning("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    elif args.device == 'xpu':
        if check_xpu_available():
            device = torch.device('xpu')
        else:
            logging.warning("XPU not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
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
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if transformer_results:
            all_results['Transformer'] = transformer_results

    # Evaluate FFNN
    if args.ffnn_model:
        ffnn_results = evaluate_nn_model(
            args.ffnn_model, 'FFNN',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if ffnn_results:
            all_results['FFNN'] = ffnn_results

    # Evaluate CNN
    if args.cnn_model:
        cnn_results = evaluate_nn_model(
            args.cnn_model, 'CNN',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if cnn_results:
            all_results['CNN'] = cnn_results

    # Evaluate CNN_Large
    if args.cnn_large_model:
        cnn_large_results = evaluate_nn_model(
            args.cnn_large_model, 'CNN_Large',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if cnn_large_results:
            all_results['CNN_Large'] = cnn_large_results

    # Evaluate ViT
    if args.vit_model:
        vit_results = evaluate_nn_model(
            args.vit_model, 'ViT',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if vit_results:
            all_results['ViT'] = vit_results

    # Evaluate ViT_Large
    if args.vit_large_model:
        vit_large_results = evaluate_nn_model(
            args.vit_large_model, 'ViT_Large',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if vit_large_results:
            all_results['ViT_Large'] = vit_large_results

    # Evaluate QubitCentric
    if args.qubit_centric_model:
        qc_results = evaluate_nn_model(
            args.qubit_centric_model, 'QUBIT_CENTRIC',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if qc_results:
            all_results['QubitCentric'] = qc_results

    # Evaluate LUT_Residual
    if args.lut_residual_model:
        lut_res_results = evaluate_nn_model(
            args.lut_residual_model, 'LUT_RESIDUAL',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if lut_res_results:
            all_results['LUT_Residual'] = lut_res_results

    # Evaluate LUT_Concat
    if args.lut_concat_model:
        lut_concat_results = evaluate_nn_model(
            args.lut_concat_model, 'LUT_CONCAT',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if lut_concat_results:
            all_results['LUT_Concat'] = lut_concat_results

    # Evaluate Diamond CNN
    if args.diamond_model:
        diamond_results = evaluate_nn_model(
            args.diamond_model, 'DIAMOND',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if diamond_results:
            all_results['Diamond'] = diamond_results

    # Evaluate Diamond Deep CNN
    if args.diamond_deep_model:
        diamond_deep_results = evaluate_nn_model(
            args.diamond_deep_model, 'DIAMOND_DEEP',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if diamond_deep_results:
            all_results['Diamond_Deep'] = diamond_deep_results

    # Evaluate ViT_QubitCentric
    if args.vit_qubit_centric_model:
        vit_qc_results = evaluate_nn_model(
            args.vit_qubit_centric_model, 'VIT_QUBIT_CENTRIC',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if vit_qc_results:
            all_results['ViT_QubitCentric'] = vit_qc_results

    # Evaluate ViT_LUT_Concat
    if args.vit_lut_concat_model:
        vit_lut_results = evaluate_nn_model(
            args.vit_lut_concat_model, 'VIT_LUT_CONCAT',
            Hx, Hz, Lx, Lz, args.p_errors,
            n_shots=args.n_shots, y_ratio=args.y_ratio, device=device,
            batch_size=args.batch_size
        )
        if vit_lut_results:
            all_results['ViT_LUT_Concat'] = vit_lut_results

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
                        default=[0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13],
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
    parser.add_argument('--cnn_model', type=str, default=None,
                        help='Path to trained CNN model')
    parser.add_argument('--cnn_large_model', type=str, default=None,
                        help='Path to trained CNN_Large model')
    parser.add_argument('--vit_model', type=str, default=None,
                        help='Path to trained ViT model')
    parser.add_argument('--vit_large_model', type=str, default=None,
                        help='Path to trained ViT_Large model')
    parser.add_argument('--qubit_centric_model', type=str, default=None,
                        help='Path to trained QubitCentric model')
    parser.add_argument('--lut_residual_model', type=str, default=None,
                        help='Path to trained LUT_Residual model')
    parser.add_argument('--lut_concat_model', type=str, default=None,
                        help='Path to trained LUT_Concat model')
    parser.add_argument('--diamond_model', type=str, default=None,
                        help='Path to trained Diamond CNN model')
    parser.add_argument('--diamond_deep_model', type=str, default=None,
                        help='Path to trained Diamond Deep CNN model')
    parser.add_argument('--vit_qubit_centric_model', type=str, default=None,
                        help='Path to trained ViT_QubitCentric model')
    parser.add_argument('--vit_lut_concat_model', type=str, default=None,
                        help='Path to trained ViT_LUT_Concat model')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'xpu', 'auto'],
                        help='Device to use (cpu, cuda, xpu, or auto for auto-detection)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for GPU/XPU inference (default: 1024)')

    args = parser.parse_args()
    main(args)

"""
Compare different decoders: MWPM vs Neural Network models
(Supports HQMT, Jung CNN, ViT_LUT_CONCAT with Measurement Error)
"""
import argparse
import os
import torch
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from qec.core.codes import Get_surface_Code
from qec.decoders.mwpm import MWPM_Decoder
from qec.training.common import (
    create_surface_code_pure_error_lut,
    simple_decoder_C_torch
)

class Code:
    pass

def get_experiment_dir(L, y_ratio, p_meas=0.0):
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
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return log_file

def _reset_seed_for_idx(idx, base_seed=20_000_000):
    seed = base_seed + idx
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_noise_batch(n_phys, p, y_ratio, batch_size, device='cpu'):
    if y_ratio > 0:
        p_y = p * y_ratio
        p_xz = p * (1 - y_ratio) / 2
        rand_vals = torch.rand(batch_size, n_phys, device=device)
        e_y = rand_vals < p_y
        e_x_only = (p_y <= rand_vals) & (rand_vals < p_y + p_xz)
        e_z_only = (p_y + p_xz <= rand_vals) & (rand_vals < p_y + 2*p_xz)
        e_x = (e_y | e_x_only).to(torch.uint8)
        e_z = (e_y | e_z_only).to(torch.uint8)
    else:
        rand_vals = torch.rand(batch_size, n_phys, device=device)
        e_z = (rand_vals < p/3).to(torch.uint8)
        e_x = ((p/3 <= rand_vals) & (rand_vals < 2*p/3)).to(torch.uint8)
        e_y = ((2*p/3 <= rand_vals) & (rand_vals < p)).to(torch.uint8)
        e_z = (e_z + e_y) % 2
        e_x = (e_x + e_y) % 2
    return e_x, e_z

def evaluate_mwpm(Hx, Hz, Lx, Lz, p_errors, n_shots=10000, y_ratio=0.0):
    logging.info("\n" + "="*60)
    logging.info("MWPM Decoder Evaluation")
    logging.info("="*60)
    decoder = MWPM_Decoder(Hx, Hz, Lx, Lz)
    results = {}
    for idx, p in enumerate(p_errors):
        _reset_seed_for_idx(idx)
        logging.info(f"\nTesting p={p:.3f}...")
        result = decoder.evaluate(p, n_shots=n_shots, y_ratio=y_ratio, verbose=True)
        result['batch_latency'] = result['avg_latency'] * n_shots
        results[p] = result
        logging.info(f"  LER: {result['ler']:.6e}")
        logging.info(f"  Avg Latency: {result['avg_latency']:.6f} ms")
    return results

def evaluate_nn_model(model_path, model_type, Hx, Hz, Lx, Lz, p_errors,
                      n_shots=10000, y_ratio=0.0, p_meas=0.0, device='cuda', batch_size=1024):
    logging.info("\n" + "="*60)
    logging.info(f"{model_type.upper()} Model Evaluation")
    logging.info("="*60)
    logging.info(f"Model path: {model_path}")
    if p_meas > 0:
        logging.info(f"Simulating Measurement Error: p_meas={p_meas} (Stacking)")

    x_error_basis = create_surface_code_pure_error_lut(int(np.sqrt(Hx.shape[1])), 'X_only', device)
    z_error_basis = create_surface_code_pure_error_lut(int(np.sqrt(Hx.shape[1])), 'Z_only', device)

    code = Code()
    code.H_x = torch.from_numpy(Hx).long().to(device)
    code.H_z = torch.from_numpy(Hz).long().to(device)
    code.L_x = torch.from_numpy(Lx).long().to(device)
    code.L_z = torch.from_numpy(Lz).long().to(device)
    code.code_type = 'surface'

    class Args: pass
    args = Args()
    args.y_ratio = y_ratio
    args.code = code
    args.no_g = 1
    args.p_meas = p_meas
    args.code_L = int(np.sqrt(Hx.shape[1]))
    # Defaults for HQMT/Jung
    args.d_model = 128
    args.n_heads = 4
    args.dim_feedforward = 512
    args.n_layers_stage1 = 2
    args.n_layers_stage2 = 4
    args.use_pos_enc = 1

    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return None

    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        logging.info("Model loaded successfully (full model)")
        
        # [Legacy Patch] Jung CNN
        if model_type.upper() == 'JUNG_CNN':
            if not hasattr(model, 'input_depth'):
                model.input_depth = model.conv1.in_channels // 2 if hasattr(model, 'conv1') else 1
                if not hasattr(model, 'grid_size_h'):
                    model.grid_size_h = args.code_L + 1
                    model.grid_size_w = args.code_L + 1
        
        if hasattr(model, 'args'):
            model.args.p_meas = p_meas

    except:
        logging.info("Loading state_dict...")
        try:
            if model_type.upper() == 'FFNN':
                from qec.models.ffnn import ECC_FFNN; model = ECC_FFNN(args, dropout=0)
            elif model_type.upper() == 'CNN':
                from qec.models.cnn import ECC_CNN; model = ECC_CNN(args, dropout=0)
            elif model_type.upper() == 'CNN_LARGE':
                from qec.models.cnn import ECC_CNN_Large; model = ECC_CNN_Large(args, dropout=0)
            elif model_type.upper() == 'VIT':
                from qec.models.vit import ECC_ViT; model = ECC_ViT(args, dropout=0)
            elif model_type.upper() == 'VIT_LARGE':
                from qec.models.vit import ECC_ViT_Large; model = ECC_ViT_Large(args, dropout=0)
            elif model_type.upper() == 'QUBIT_CENTRIC':
                from qec.models.qubit_centric import ECC_QubitCentric; model = ECC_QubitCentric(args, dropout=0)
            elif model_type.upper() == 'LUT_RESIDUAL':
                from qec.models.qubit_centric import ECC_LUT_Residual; model = ECC_LUT_Residual(args, x_error_basis, z_error_basis, dropout=0)
            elif model_type.upper() == 'LUT_CONCAT':
                from qec.models.qubit_centric import ECC_LUT_Concat; model = ECC_LUT_Concat(args, x_error_basis, z_error_basis, dropout=0)
            elif model_type.upper() == 'DIAMOND':
                from qec.models.diamond_cnn import ECC_DiamondCNN; model = ECC_DiamondCNN(args, x_error_lut=x_error_basis, z_error_lut=z_error_basis, dropout=0)
            elif model_type.upper() == 'DIAMOND_DEEP':
                from qec.models.diamond_cnn import ECC_DiamondCNN_Deep; model = ECC_DiamondCNN_Deep(args, x_error_lut=x_error_basis, z_error_lut=z_error_basis, dropout=0)
            elif model_type.upper() == 'VIT_QUBIT_CENTRIC':
                from qec.models.vit import ECC_ViT_QubitCentric; model = ECC_ViT_QubitCentric(args, dropout=0)
            elif model_type.upper() == 'VIT_LUT_CONCAT':
                from qec.models.vit import ECC_ViT_LUT_Concat; model = ECC_ViT_LUT_Concat(args, x_error_basis, z_error_basis, dropout=0)
            # New Models
            elif model_type.upper() == 'JUNG_CNN':
                from qec.models.jung_cnn import JungCNNDecoder; model = JungCNNDecoder(args, dropout=0, label_smoothing=0)
            elif model_type.upper() == 'HQMT':
                from qec.models.hqmt import HQMT; model = HQMT(args, x_error_lut=x_error_basis, z_error_lut=z_error_basis, dropout=0)
            else:
                logging.error(f"Unknown model type: {model_type}"); return None

            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            from collections import OrderedDict
            new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
            model.load_state_dict(new_state_dict)
            model.eval()
        except Exception as e2:
            logging.error(f"Failed to load model as state_dict: {e2}"); return None

    model.to(device)
    results = {}
    n_phys = Hx.shape[1]
    
    use_batch = device.type in ['cuda', 'xpu']
    if not use_batch: batch_size = 1
    
    # 2D Model Check
    is_2d_model = model_type.upper() not in ['HQMT', 'JUNG_CNN']

    with torch.no_grad():
        for idx, p in enumerate(p_errors):
            _reset_seed_for_idx(idx)
            logging.info(f"\nTesting p={p:.3f}...")
            logical_errors = 0
            total_time = 0
            n_batches = (n_shots + batch_size - 1) // batch_size

            for i in tqdm(range(n_batches)):
                curr_bs = min(batch_size, n_shots - i * batch_size)
                e_x, e_z = generate_noise_batch(n_phys, p, y_ratio, curr_bs, device)
                e_full = torch.cat([e_z, e_x], dim=1)
                
                s_z = (code.H_z.float() @ e_x.float().T).T % 2
                s_x = (code.H_x.float() @ e_z.float().T).T % 2
                syn_perfect = torch.cat([s_z, s_x], dim=1)

                if p_meas > 0:
                    noisy_frames = [(syn_perfect + (torch.rand_like(syn_perfect)<p_meas).float())%2 for _ in range(args.code_L+1)]
                    syn_input = torch.stack(noisy_frames, dim=1)
                else:
                    syn_input = syn_perfect

                if is_2d_model and p_meas > 0:
                    model_input = syn_input[:, -1, :]
                else:
                    model_input = syn_input

                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model(model_input.float())
                if isinstance(out, tuple): out = out[0]
                _, pred = torch.max(out.data, 1)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                total_time += (time.perf_counter() - t0)

                for j in range(curr_bs):
                    syn_idx = syn_perfect[j].type(torch.uint8)
                    pure_err = simple_decoder_C_torch(syn_idx, x_error_basis, z_error_basis, code.H_z, code.H_x)
                    l_phys = pure_err.long() ^ e_full[j].long()
                    l_x_flip = (code.L_z.float() @ l_phys[n_phys:].float()) % 2
                    l_z_flip = (code.L_x.float() @ l_phys[:n_phys].float()) % 2
                    true_cls = (l_z_flip * 2 + l_x_flip).long()
                    if pred[j].item() != true_cls.item(): logical_errors += 1

            ler = logical_errors / n_shots
            avg_latency = (total_time * 1000) / n_shots
            results[p] = {'ler': ler, 'avg_latency': avg_latency}
            logging.info(f"  LER: {ler:.6e}")
            logging.info(f"  Avg Latency: {avg_latency:.6f} ms")
    return results

def print_comparison_table(results_dict):
    logging.info("\n" + "="*80)
    logging.info("COMPARISON SUMMARY")
    logging.info("="*80)
    p_values = sorted(list(next(iter(results_dict.values())).keys()))
    header = f"{'p_error':<12}" + "".join([f"{name + ' LER':<20}" for name in results_dict.keys()])
    logging.info(header); logging.info("-" * len(header))
    for p in p_values:
        row = f"{p:<12.3f}"
        for name in results_dict.keys():
            if p in results_dict[name]: row += f"{results_dict[name][p]['ler']:<20.6e}"
            else: row += f"{'N/A':<20}"
        logging.info(row)
    logging.info("="*80)

def plot_comparison_graphs(results_dict, save_dir, L, y_ratio):
    if not results_dict: return
    p_values = sorted(list(next(iter(results_dict.values())).keys()))
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    styles = {
        'MWPM': {'marker': 'o', 'color': 'black'},
        'Transformer': {'marker': 's', 'color': '#1f77b4'},
        'FFNN': {'marker': '^', 'color': '#d62728'},
        'CNN': {'marker': 'd', 'color': '#2ca02c'},
        'ViT': {'marker': 'p', 'color': '#9467bd'},
        'Jung_CNN': {'marker': 'X', 'color': '#000080', 'linestyle': '--', 'linewidth': 2},
        'HQMT': {'marker': '*', 'color': '#FF1493', 'linestyle': '-.', 'linewidth': 2},
        'ViT_LUT_CONCAT': {'marker': 's', 'color': 'green', 'linestyle': ':'}
    }

    for name, res in results_dict.items():
        lers = [res[p]['ler'] for p in p_values]
        style = styles.get(name, {'marker': 'x'})
        ax1.plot(p_values, lers, label=name, **style)

    ax1.set_xlabel('Physical Error Rate (p)')
    ax1.set_ylabel('Logical Error Rate (LER)')
    ax1.set_title(f'LER Comparison (L={L})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log'); ax1.set_xscale('log')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f'comparison_plot_{timestamp}.png')
    plt.savefig(path, dpi=300)
    logging.info(f"\nPlot saved to: {path}")
    plt.close(fig)

def main(args):
    EVAL_SEED = 20_000_000
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(EVAL_SEED)

    exp_dir = get_experiment_dir(args.L, args.y_ratio, args.p_meas)
    setup_logging(exp_dir)
    logging.info(f"Decoder Comparison Script (L={args.L}, p_meas={args.p_meas})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device != 'auto': device = torch.device(args.device)
    
    Hx, Hz, Lx, Lz = Get_surface_Code(args.L)
    all_results = {}

    if not args.skip_mwpm:
        all_results['MWPM'] = evaluate_mwpm(Hx, Hz, Lx, Lz, args.p_errors, args.n_shots, args.y_ratio)

    def run(path, name):
        if path:
            res = evaluate_nn_model(path, name, Hx, Hz, Lx, Lz, args.p_errors, 
                                  args.n_shots, args.y_ratio, args.p_meas, device, args.batch_size)
            if res: all_results[name] = res

    # Original Models
    run(args.transformer_model, 'Transformer')
    run(args.ffnn_model, 'FFNN')
    run(args.cnn_model, 'CNN')
    run(args.cnn_large_model, 'CNN_Large')
    run(args.vit_model, 'ViT')
    run(args.vit_large_model, 'ViT_Large')
    run(args.qubit_centric_model, 'QubitCentric')
    run(args.lut_residual_model, 'LUT_Residual')
    run(args.lut_concat_model, 'LUT_Concat')
    run(args.diamond_model, 'Diamond')
    run(args.diamond_deep_model, 'Diamond_Deep')
    run(args.vit_qubit_centric_model, 'ViT_QubitCentric')
    
    # New Models (Added)
    run(args.vit_lut_concat_model, 'ViT_LUT_CONCAT')
    run(args.jung_model, 'Jung_CNN')
    run(args.hqmt_model, 'HQMT')

    if all_results:
        print_comparison_table(all_results)
        plot_comparison_graphs(all_results, exp_dir, args.L, args.y_ratio)
    else:
        logging.error("No results to compare!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare QEC Decoders')
    parser.add_argument('-L', type=int, default=3)
    # [원상복구] Original Default Values
    parser.add_argument('-p', '--p_errors', type=float, nargs='+',
                        default=[0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13])
    parser.add_argument('-n', '--n_shots', type=int, default=10000)
    parser.add_argument('-y', '--y_ratio', type=float, default=0.0)
    
    # [추가] Measurement Error
    parser.add_argument('--p_meas', type=float, default=0.0)

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--skip_mwpm', action='store_true')

    # Models
    parser.add_argument('--transformer_model', type=str)
    parser.add_argument('--ffnn_model', type=str)
    parser.add_argument('--cnn_model', type=str)
    parser.add_argument('--cnn_large_model', type=str)
    parser.add_argument('--vit_model', type=str)
    parser.add_argument('--vit_large_model', type=str)
    parser.add_argument('--qubit_centric_model', type=str)
    parser.add_argument('--lut_residual_model', type=str)
    parser.add_argument('--lut_concat_model', type=str)
    parser.add_argument('--diamond_model', type=str)
    parser.add_argument('--diamond_deep_model', type=str)
    parser.add_argument('--vit_qubit_centric_model', type=str)
    
    # [추가] New Models
    parser.add_argument('--vit_lut_concat_model', type=str)
    parser.add_argument('--jung_model', type=str)
    parser.add_argument('--hqmt_model', type=str)

    args = parser.parse_args()
    main(args)
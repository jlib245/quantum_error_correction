"""
Train Transformer with Stim-generated syndromes

Supports both dual-observable (4-class) and single-observable (binary) modes.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import stim
import argparse
from datetime import datetime
import os
import logging
import time

from qec.models.transformer import ECC_Transformer
from qec.core.stim_circuits import create_rotated_surface_code_circuit

# Performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Define these classes at module level for pickling
class ModelArgs:
    """Model arguments for ECC_Transformer"""
    pass


class Code:
    """Dummy code object for model initialization"""
    pass


class StimSurfaceCodeDataset(Dataset):
    """Dataset using Stim for syndrome generation with dual observables (4-class)"""

    def __init__(self, distance: int, error_rates: list, samples_per_epoch: int,
                 rounds: int = 1, y_ratio: float = 0.0):
        self.distance = distance
        self.error_rates = error_rates
        self.samples_per_epoch = samples_per_epoch
        self.rounds = rounds
        self.y_ratio = y_ratio

        # Create Stim circuits for each error rate
        self.circuits = {}
        self.samplers = {}

        for p in error_rates:
            # Use custom circuit with both observables (4-class)
            circuit = create_rotated_surface_code_circuit(
                distance=distance,
                rounds=rounds,
                p_error=p,
                y_ratio=y_ratio
            )
            self.circuits[p] = circuit
            self.samplers[p] = circuit.compile_detector_sampler()

        # Get syndrome dimension
        self.syndrome_dim = self.circuits[error_rates[0]].num_detectors
        self.num_observables = self.circuits[error_rates[0]].num_observables

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Random error rate
        p = np.random.choice(self.error_rates)

        # Sample from Stim
        detection, observable = self.samplers[p].sample(1, separate_observables=True)

        syndrome = detection[0].astype(np.float32)

        # 4-class: combine both observables
        x_flip = int(observable[0, 0])
        z_flip = int(observable[0, 1])
        true_class = z_flip * 2 + x_flip  # 0=I, 1=X, 2=Z, 3=Y

        return torch.from_numpy(syndrome), torch.tensor(true_class, dtype=torch.long)


def train_with_stim(args):
    """Train Transformer with Stim-generated data"""

    # Setup device
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    elif args.device == 'xpu':
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = torch.device('xpu')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    # Create dataset
    train_dataset = StimSurfaceCodeDataset(
        distance=args.code_L,
        error_rates=args.p_errors,
        samples_per_epoch=args.samples_per_epoch,
        rounds=args.rounds,
        y_ratio=args.y_ratio
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if device.type in ['cuda', 'xpu'] else False,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=4 if args.workers > 0 else None
    )

    # Create model
    model_args = ModelArgs()
    model_args.d_model = args.d_model
    model_args.h = args.h
    model_args.N_dec = args.N_dec
    model_args.no_mask = args.no_mask
    model_args.no_g = args.no_g

    # Dummy code object for model initialization
    code = Code()
    code.pc_matrix = torch.zeros(train_dataset.syndrome_dim, 1)  # Dummy

    model_args.code = code

    model = ECC_Transformer(model_args, dropout=0.1).to(device)

    # Compile model for faster execution (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device.type == 'cuda':
        model = torch.compile(model, mode='reduce-overhead')
        logging.info("Model compiled with torch.compile()")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Mixed Precision Training
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Training loop
    model_dir = f"Stim_Models/L{args.code_L}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_dir, exist_ok=True)

    # Setup logging to file
    log_file = os.path.join(model_dir, 'logging.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ],
        force=True  # Override any existing config
    )

    # Log initial info
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(f"Arguments: {args}")
    logging.info(f"Using device: {device}")
    logging.info(f"Stim Dataset: Distance={args.code_L}, Rounds={args.rounds}")
    logging.info(f"  Syndrome_dim={train_dataset.syndrome_dim}, Num_classes=4")
    logging.info(f"  Y_ratio={args.y_ratio}")
    logging.info(f"Samples per epoch: {args.samples_per_epoch}, Batch size: {args.batch_size}")
    logging.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    best_loss = float('inf')
    patience_counter = 0

    num_batches = len(train_loader)
    print_interval = max(1, num_batches // 2)  # Print at middle and end

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        epoch_start_time = time.time()

        for batch_idx, (syndrome, labels) in enumerate(train_loader, 1):
            syndrome = syndrome.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(syndrome)
                    loss = model.loss(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(syndrome)
                loss = model.loss(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(syndrome)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += len(labels)

            # Print progress at intervals
            if batch_idx % print_interval == 0 or batch_idx == num_batches:
                avg_loss = total_loss / total
                ler = 1.0 - (correct / total)  # Logical Error Rate
                logging.info(f"Training epoch {epoch}, Batch {batch_idx}/{num_batches}: "
                             f"LR={args.lr:.2e}, Loss={avg_loss:.5e} LER={ler:.3e}")
                logging.info(f"***Loss={avg_loss:.5e}")

        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / total
        ler = 1.0 - (correct / total)

        logging.info(f"Epoch {epoch} Train Time {epoch_time}s\n")

        # Save best model and check early stopping
        if avg_loss < best_loss - args.min_delta:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model, os.path.join(model_dir, 'best_model'))
            logging.info(f"Model Saved - New best loss: {avg_loss:.5e}")
        else:
            patience_counter += 1
            logging.info(f"No improvement. Patience: {patience_counter}/{args.patience}")

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            logging.info(f"\nEarly stopping triggered after {epoch} epochs (patience={args.patience})")
            break

    logging.info(f"\nTraining complete! Model saved to: {model_dir}")
    logging.info(f"Best loss: {best_loss:.5e}")
    return model_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--code_L', type=int, default=5,
                        help='Code distance')
    parser.add_argument('--rounds', type=int, default=5,
                        help='QEC rounds (default: 5, recommended: L)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum training epochs')
    parser.add_argument('--samples_per_epoch', type=int, default=100000,
                        help='Number of samples per epoch (default: 100000)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda', 'xpu'],
                        help='Device to use')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer model dimension')
    parser.add_argument('--h', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--N_dec', type=int, default=10,
                        help='Number of decoder layers')
    parser.add_argument('-p', '--p_errors', type=float, nargs='+',
                        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11],
                        help='Physical error rates for training')
    parser.add_argument('-y', '--y_ratio', type=float, default=0.0,
                        help='Ratio of Y errors (0.0 = standard depolarizing)')
    parser.add_argument('--patience', type=int, default=40,
                        help='Early stopping patience (0 = disabled)')
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='Minimum loss improvement for early stopping')
    parser.add_argument('--no_g', type=int, default=0,
                        help='Disable gauge fixing (default: 0=enabled)')
    parser.add_argument('--no_mask', type=int, default=0,
                        help='Disable masking (default: 0=enabled)')

    args = parser.parse_args()

    train_with_stim(args)

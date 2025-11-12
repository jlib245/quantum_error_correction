"""
Train Transformer with Stim-generated syndromes

This ensures perfect compatibility between training and Stim evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import stim
import argparse
from datetime import datetime
import os
import logging

from qec.models.transformer import ECC_Transformer


class StimSurfaceCodeDataset(Dataset):
    """Dataset using Stim for syndrome generation"""

    def __init__(self, distance: int, error_rates: list, samples_per_epoch: int):
        self.distance = distance
        self.error_rates = error_rates
        self.samples_per_epoch = samples_per_epoch

        # Create Stim circuits for each error rate
        self.circuits = {}
        self.samplers = {}

        for p in error_rates:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                distance=distance,
                rounds=1,
                after_clifford_depolarization=p
            )
            self.circuits[p] = circuit
            self.samplers[p] = circuit.compile_detector_sampler()

        # Get syndrome dimension
        self.syndrome_dim = self.circuits[error_rates[0]].num_detectors

        print(f"Stim Dataset Created:")
        print(f"  Distance: {distance}")
        print(f"  Syndrome dimension: {self.syndrome_dim}")
        print(f"  Error rates: {error_rates}")
        print(f"  Samples per epoch: {samples_per_epoch}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Random error rate
        p = np.random.choice(self.error_rates)

        # Sample from Stim
        detection, observable = self.samplers[p].sample(1, separate_observables=True)

        syndrome = detection[0].astype(np.float32)
        logical_error = int(observable[0, 0])  # 0 or 1

        # Map to 4 classes (simplified: only I and X)
        # Note: Stim observable only tracks one logical operator
        true_class = logical_error  # 0=I, 1=X

        return torch.from_numpy(syndrome), torch.tensor(true_class, dtype=torch.long)


def train_with_stim(args):
    """Train Transformer with Stim-generated data"""

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset
    train_dataset = StimSurfaceCodeDataset(
        distance=args.code_L,
        error_rates=args.error_rates,
        samples_per_epoch=100000
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    # Create model
    class Args:
        pass

    model_args = Args()
    model_args.d_model = args.d_model
    model_args.h = args.h
    model_args.N_dec = args.N_dec
    model_args.no_mask = 0

    # Dummy code object for model initialization
    class Code:
        pass

    code = Code()
    code.pc_matrix = torch.zeros(train_dataset.syndrome_dim, 1)  # Dummy

    model_args.code = code

    model = ECC_Transformer(model_args, dropout=0).to(device)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    model_dir = f"Stim_Models/L{args.code_L}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_dir, exist_ok=True)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for syndrome, labels in train_loader:
            syndrome = syndrome.to(device)
            labels = labels.to(device)

            outputs = model(syndrome)
            loss = model.loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(syndrome)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / total
        accuracy = correct / total

        print(f"Epoch {epoch}/{args.epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")

        # Save best model and check early stopping
        if avg_loss < best_loss - args.min_delta:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model, os.path.join(model_dir, 'best_model'))
            print(f"  → Saved best model (loss={avg_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (patience={args.patience})")
            break

    print(f"\nTraining complete! Model saved to: {model_dir}")
    print(f"Best loss: {best_loss:.4f}")
    return model_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--code_L', type=int, default=3,
                        help='Code distance')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda', 'xpu'],
                        help='Device to use')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer model dimension')
    parser.add_argument('--h', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--N_dec', type=int, default=6,
                        help='Number of decoder layers')
    parser.add_argument('--error_rates', type=float, nargs='+',
                        default=[0.07, 0.08, 0.09, 0.10, 0.11],
                        help='Error rates for training')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (0 = disabled)')
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='Minimum loss improvement for early stopping')

    args = parser.parse_args()

    train_with_stim(args)

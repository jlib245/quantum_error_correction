"""
Circuit-Level Noise Training Script for Surface Code

Uses Stim for realistic circuit-level noise simulation including:
- Gate errors (2-qubit depolarizing)
- Measurement errors
- Preparation/reset errors
- Idle errors

Models work with detector syndromes from Stim.
"""

import argparse
import random
import os
from torch.utils.data import DataLoader
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from qec.data.circuit_level import CircuitLevelDataset, create_circuit_level_surface_code
from qec.training.common import setup_device, save_checkpoint, load_checkpoint


class CircuitLevelCNN(nn.Module):
    """
    CNN decoder for circuit-level noise.

    Takes detector syndrome from Stim and outputs 4-class prediction.
    Architecture adapts to different code distances and rounds.
    """

    def __init__(self, num_detectors, d_model=128, dropout=0.2, label_smoothing=0.1):
        super().__init__()
        self.num_detectors = num_detectors
        self.label_smoothing = label_smoothing

        # Input projection
        self.input_proj = nn.Linear(num_detectors, d_model)

        # Residual blocks
        self.blocks = nn.Sequential(
            ResBlock(d_model, dropout),
            ResBlock(d_model, dropout),
            ResBlock(d_model, dropout),
            ResBlock(d_model, dropout),
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4)  # 4 classes: I, X, Z, Y
        )

    def forward(self, syndrome, labels=None):
        """
        Args:
            syndrome: (batch, num_detectors) detector syndrome
            labels: (batch,) optional labels for training

        Returns:
            logits: (batch, 4) class logits
            loss: scalar loss if labels provided
        """
        x = self.input_proj(syndrome)
        x = self.blocks(x)
        logits = self.output_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits, labels,
                label_smoothing=self.label_smoothing
            )

        return logits, loss


class ResBlock(nn.Module):
    """Residual block with pre-norm."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class CircuitLevelConv2D(nn.Module):
    """
    2D CNN decoder that reshapes detector syndrome into spatial format.

    For d rounds of syndrome measurement, detector syndrome can be viewed as
    a 2D+T tensor of shape (rounds, num_stabilizers).
    """

    def __init__(self, distance, rounds, d_model=64, dropout=0.2, label_smoothing=0.1):
        super().__init__()
        self.distance = distance
        self.rounds = rounds
        self.label_smoothing = label_smoothing

        # Number of stabilizers per round
        # For rotated surface code: (d-1)^2 + boundary stabilizers
        self.num_x_stab = (distance - 1) ** 2 // 2 + (distance - 1)
        self.num_z_stab = (distance - 1) ** 2 - (distance - 1) ** 2 // 2 + (distance - 1)
        self.stab_per_round = self.num_x_stab + self.num_z_stab

        # Total detectors = rounds * stab_per_round
        self.total_detectors = rounds * self.stab_per_round

        # Spatial size for reshaping
        self.spatial_size = distance  # Approximate as d x d grid

        # 2D CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(rounds * 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
        )

        # Global pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4)
        )

    def forward(self, syndrome, labels=None):
        batch_size = syndrome.shape[0]

        # Reshape to (batch, rounds*2, d, d) approximately
        # Split into X and Z stabilizer channels
        x = syndrome.view(batch_size, self.rounds, -1)

        # Pad to square grid
        target_size = self.distance * self.distance
        if x.shape[-1] < target_size:
            padding = target_size - x.shape[-1]
            x = F.pad(x, (0, padding))

        x = x.view(batch_size, self.rounds, self.distance, self.distance)

        # Duplicate channels for X and Z (simplified)
        x = x.repeat(1, 2, 1, 1)

        x = self.conv_layers(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        return logits, loss


def train_epoch(model, device, train_loader, optimizer, epoch, lr):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (syndrome, labels) in enumerate(train_loader):
        syndrome = syndrome.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, loss = model(syndrome, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 100 == 0:
            logging.info(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                        f'Loss: {loss.item():.4f} Acc: {correct/total:.4f} LR: {lr:.6f}')

    return total_loss / len(train_loader)


def evaluate(model, device, dataset, error_rate, n_samples=10000):
    """Evaluate model at a specific error rate."""
    model.eval()

    syndromes, labels = dataset.sample_batch(n_samples, p=error_rate)
    syndromes = syndromes.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits, _ = model(syndromes)
        pred = logits.argmax(dim=1)

        # Logical error = any wrong prediction
        errors = (pred != labels).sum().item()
        ler = errors / n_samples

    return ler


def main(args):
    device = setup_device(args.device)

    # Create dataset
    train_dataset = CircuitLevelDataset(
        distance=args.code_L,
        error_rates=args.p_errors,
        length=args.samples_per_epoch,
        rounds=args.rounds,
        y_ratio=args.y_ratio,
        seed=args.seed
    )

    # Test dataset (separate seed)
    test_dataset = CircuitLevelDataset(
        distance=args.code_L,
        error_rates=args.p_errors,
        length=args.test_samples,
        rounds=args.rounds,
        y_ratio=args.y_ratio,
        seed=args.seed + 1000000
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Create model
    num_detectors = train_dataset.num_detectors
    logging.info(f'Number of detectors: {num_detectors}')

    if args.model_type == 'mlp':
        model = CircuitLevelCNN(
            num_detectors=num_detectors,
            d_model=args.d_model,
            dropout=args.dropout,
            label_smoothing=args.label_smoothing
        ).to(device)
    elif args.model_type == 'conv2d':
        model = CircuitLevelConv2D(
            distance=args.code_L,
            rounds=args.rounds,
            d_model=args.d_model,
            dropout=args.dropout,
            label_smoothing=args.label_smoothing
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    logging.info(f'Code L={args.code_L}, Rounds={args.rounds}')
    logging.info(f'Model type: {args.model_type}')
    logging.info(f'Y-ratio: {args.y_ratio}')
    logging.info(model)
    logging.info(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Training loop
    best_val_ler = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, device, train_loader, optimizer, epoch,
                                  scheduler.get_last_lr()[0])
        scheduler.step()

        # Validation
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            val_lers = []
            for p in args.p_errors:
                ler = evaluate(model, device, test_dataset, p, args.test_samples)
                val_lers.append(ler)
                logging.info(f'  p={p:.3f}: LER={ler:.5e}')

            mean_val_ler = np.mean(val_lers)
            logging.info(f'Epoch {epoch}: Mean Val LER = {mean_val_ler:.5e}')

            # Save best model
            if mean_val_ler < best_val_ler - args.min_delta:
                best_val_ler = mean_val_ler
                patience_counter = 0

                # Save model
                model_path = os.path.join(args.path, 'best_model.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_ler': best_val_ler,
                    'args': args,
                }, model_path)

                # Also save full model for easy loading
                full_model_path = os.path.join(args.path, 'best_model.zip')
                torch.save(model, full_model_path)

                logging.info(f'Model Saved - Best val LER: {best_val_ler:.5e}')
            else:
                patience_counter += 1
                logging.info(f'No improvement. Patience: {patience_counter}/{args.patience}')

            # Early stopping
            if args.patience > 0 and patience_counter >= args.patience:
                logging.info(f'Early stopping at epoch {epoch}')
                break

    logging.info(f'Training complete. Best Val LER: {best_val_ler:.5e}')
    return best_val_ler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Circuit-Level Noise Decoder Training')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--test_samples', type=int, default=10000)
    parser.add_argument('--samples_per_epoch', type=int, default=500000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=0.0)
    parser.add_argument('--val_interval', type=int, default=1)

    # Code
    parser.add_argument('-L', '--code_L', type=int, default=5)
    parser.add_argument('-r', '--rounds', type=int, default=1,
                        help='Number of syndrome measurement rounds')
    parser.add_argument('-p', '--p_errors', type=float, nargs='+',
                        default=[0.001, 0.002, 0.003, 0.004, 0.005],
                        help='Gate error probabilities (circuit-level uses lower rates)')
    parser.add_argument('-y', '--y_ratio', type=float, default=0.0,
                        help='Y error bias ratio')

    # Model
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'conv2d'],
                        help='Model type: mlp or conv2d')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    args = parser.parse_args()

    # Setup output dir
    timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    model_name = f'circuit_level_{args.model_type.upper()}'

    args.path = f'Final_Results/surface/L_{args.code_L}/circuit_level/rounds_{args.rounds}/y_{args.y_ratio}/{model_name}/{timestamp}'
    os.makedirs(args.path, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.path, 'logging.txt')),
            logging.StreamHandler()
        ]
    )
    logging.info(f'Args: {args}')

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)

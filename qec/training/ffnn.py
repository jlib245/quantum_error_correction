"""
FFNN Training Script for Quantum Error Correction
"""
import argparse
import random
import os
from torch.utils.data import DataLoader
from datetime import datetime
import logging
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from qec.training.common import (
    QECC_Dataset,
    setup_device,
    load_surface_code,
    create_surface_code_pure_error_lut,
    train_epoch,
    test_model,
    save_checkpoint,
)


def main(args):
    device = setup_device(args.device)

    # Load code
    code = load_surface_code(args.code_L, device)
    args.code = code

    x_error_basis_dict = create_surface_code_pure_error_lut(args.code_L, 'X_only', device)
    z_error_basis_dict = create_surface_code_pure_error_lut(args.code_L, 'Z_only', device)

    # Create model
    from qec.models.ffnn import ECC_FFNN, ECC_FFNN_Large

    if args.large:
        model = ECC_FFNN_Large(args, dropout=args.dropout, label_smoothing=args.label_smoothing).to(device)
    else:
        model = ECC_FFNN(args, dropout=args.dropout, label_smoothing=args.label_smoothing).to(device)

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    logging.info(f'Code L={args.code_L}')
    logging.info(model)
    logging.info(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    ps_train = args.p_errors
    ps_test = args.p_errors

    # Dataloaders
    train_loader = DataLoader(
        QECC_Dataset(code, x_error_basis_dict, z_error_basis_dict, ps_train,
                    length=args.samples_per_epoch, args=args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
    )

    test_loaders = [
        DataLoader(
            QECC_Dataset(code, x_error_basis_dict, z_error_basis_dict, [p],
                        length=args.test_batch_size, args=args, seed_offset=10_000_000),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=True,
        ) for p in ps_test
    ]

    # Training loop
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, device, train_loader, optimizer, epoch, scheduler.get_last_lr()[0])
        scheduler.step()

        if loss < best_loss - args.min_delta:
            best_loss = loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_loss, args.path)
            logging.info(f'Model Saved - Best loss: {best_loss:.5e}')
        else:
            patience_counter += 1
            logging.info(f'No improvement. Patience: {patience_counter}/{args.patience}')

        if args.patience > 0 and patience_counter >= args.patience:
            logging.info(f'Early stopping at epoch {epoch}')
            break

        if epoch % 10 == 0:
            test_model(model, device, test_loaders, ps_test, args.test_samples)

    # Final test
    model = torch.load(os.path.join(args.path, 'best_model'), weights_only=False).to(device)
    test_model(model, device, test_loaders, ps_test, args.test_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFNN Decoder Training')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--test_samples', type=int, default=10000)
    parser.add_argument('--samples_per_epoch', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min_delta', type=float, default=0.0)

    # Code
    parser.add_argument('-L', '--code_L', type=int, default=5)
    parser.add_argument('-p', '--p_errors', type=float, nargs='+',
                        default=[0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.11, 0.12, 0.13])
    parser.add_argument('-y', '--y_ratio', type=float, default=0.0)

    # Model
    parser.add_argument('--large', action='store_true', help='Use larger FFNN')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    args = parser.parse_args()

    # Setup output dir
    timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    model_name = 'FFNN_Large' if args.large else 'FFNN'
    args.path = f'Final_Results_FFNN/surface/{model_name}_L_{args.code_L}/{timestamp}'
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

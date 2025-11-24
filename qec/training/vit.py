"""
ViT Training Script for Quantum Error Correction
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
    train_with_validation,
)


def main(args):
    device = setup_device(args.device)

    # Load code
    code = load_surface_code(args.code_L, device)
    args.code = code
    args.L = args.code_L

    x_error_basis_dict = create_surface_code_pure_error_lut(args.code_L, 'X_only', device)
    z_error_basis_dict = create_surface_code_pure_error_lut(args.code_L, 'Z_only', device)

    # Create model
    from qec.models.vit import ECC_ViT, ECC_ViT_Large

    if args.large:
        model = ECC_ViT_Large(args, dropout=args.dropout, label_smoothing=args.label_smoothing).to(device)
    else:
        model = ECC_ViT(args, dropout=args.dropout, label_smoothing=args.label_smoothing).to(device)

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

    val_loaders = [
        DataLoader(
            QECC_Dataset(code, x_error_basis_dict, z_error_basis_dict, [p],
                        length=args.test_samples, args=args, seed_offset=10_000_000),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=True,
        ) for p in ps_test
    ]

    # Training with validation-based early stopping
    train_with_validation(model, device, train_loader, val_loaders, optimizer, scheduler,
                          args, ps_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT Decoder Training')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--test_batch_size', type=int, default=4096)
    parser.add_argument('--test_samples', type=int, default=10000)
    parser.add_argument('--samples_per_epoch', type=int, default=2000000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min_delta', type=float, default=0.0)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)

    # Code
    parser.add_argument('-L', '--code_L', type=int, default=5)
    parser.add_argument('-p', '--p_errors', type=float, nargs='+',
                        default=[0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13])
    parser.add_argument('-y', '--y_ratio', type=float, default=0.0)

    # Model
    parser.add_argument('--large', action='store_true', help='Use larger ViT')
    parser.add_argument('--N_dec', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    args = parser.parse_args()

    # Setup output dir
    timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    if args.large:
        model_name = 'ViT_Large'
    else:
        model_name = 'ViT'
    args.path = f'Final_Results/surface/L_{args.code_L}/y_{args.y_ratio}/{model_name}/{timestamp}'
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

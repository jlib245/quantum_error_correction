"""
Jung CNN Training Script for Quantum Error Correction
(Paper-compliant version with SGD and ReduceLROnPlateau)
"""
import argparse
import random
import os
from torch.utils.data import DataLoader
from datetime import datetime
import logging
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD

from qec.training.common import (
    QECC_Dataset,
    setup_device,
    load_surface_code,
    create_surface_code_pure_error_lut,
    train_epoch,
    test_model,
    save_checkpoint,
    load_checkpoint
)


def main(args):
    device = setup_device(args.device)

    # Load code
    code = load_surface_code(args.code_L, device)
    args.code = code

    x_error_basis_dict = create_surface_code_pure_error_lut(args.code_L, 'X_only', device)
    z_error_basis_dict = create_surface_code_pure_error_lut(args.code_L, 'Z_only', device)

    # Create model
    from qec.models.jung_cnn import JungCNNDecoder
    model = JungCNNDecoder(args, dropout=args.dropout, label_smoothing=args.label_smoothing).to(device)

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # [수정] Optimizer: SGD (논문 명시)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # [수정] Scheduler: ReduceLROnPlateau (논문 명시: adaptive to 10^-5)
    # min_lr 인자를 여기서 연결합니다.
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=args.scheduler_patience, 
        verbose=True,
        min_lr=args.min_lr  # <--- 추가됨
    )

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

    # Custom training loop
    start_epoch = 1
    best_val_ler = float('inf')
    
    if hasattr(args, 'resume') and args.resume and os.path.exists(args.resume):
        loaded_epoch, loaded_best = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        if loaded_epoch:
            start_epoch = loaded_epoch
            best_val_ler = loaded_best

    patience_counter = 0
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_epoch(model, device, train_loader, optimizer, epoch, optimizer.param_groups[0]['lr'])

        # Validation
        val_lers = test_model(model, device, val_loaders, ps_test, args.test_samples)
        mean_val_ler = np.mean(val_lers)
        
        # Scheduler step based on validation LER
        scheduler.step(mean_val_ler)

        # Save best model
        if mean_val_ler < best_val_ler - args.min_delta:
            best_val_ler = mean_val_ler
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_ler, args.path, metric_name='val_ler')
            logging.info(f'Model Saved - Best val LER: {best_val_ler:.5e}')
        else:
            patience_counter += 1
            logging.info(f'No improvement. Patience: {patience_counter}/{args.patience}')

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            logging.info(f'Early stopping at epoch {epoch}')
            break
            
    # Final test
    logging.info("--- Final Test with Best Model ---")
    model_path = os.path.join(args.path, 'best_model')
    if os.path.exists(model_path):
        best_model = torch.load(model_path, weights_only=False).to(device)
        test_model(best_model, device, val_loaders, ps_test, args.test_samples * 5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jung CNN Decoder Training (Paper-compliant)')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    
    # [수정] 누락되었던 min_lr 인자 추가
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate for scheduler')
    
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--test_batch_size', type=int, default=4096)
    parser.add_argument('--test_samples', type=int, default=20000)
    parser.add_argument('--samples_per_epoch', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--scheduler_patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=1e-5)
    parser.add_argument('--resume', type=str, default=None)

    # Code
    parser.add_argument('-L', '--code_L', type=int, default=5)
    parser.add_argument('-p', '--p_errors', type=float, nargs='+',
                        default=[0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13])
    parser.add_argument('-y', '--y_ratio', type=float, default=0.0)

    # Model
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    args = parser.parse_args()

    # Setup output dir
    timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    model_name = f'JungCNN_SGD' 
    args.path = f'Final_Results/surface/L_{args.code_L}/y_{args.y_ratio}/{model_name}/{timestamp}'
    os.makedirs(args.path, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
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
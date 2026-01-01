"""
Jung CNN Training Script for Quantum Error Correction
(Paper-compliant version with SGD and ReduceLROnPlateau)
Supports both Code Capacity (p_meas=0) and Phenomenological Noise (p_meas>0)
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

    # Optimizer: SGD (Paper compliant)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Scheduler: ReduceLROnPlateau (Paper compliant)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=args.scheduler_patience, 
        min_lr=args.min_lr
    )

    logging.info(f'Code L={args.code_L}')
    
    # [추가] 노이즈 모델 정보 로깅
    if args.p_meas > 0:
        logging.info(f'Noise Model: Phenomenological (p_meas={args.p_meas})')
        logging.info(f'Input Structure: Stacked 3D Volume (Depth = {args.code_L + 1})')
    else:
        logging.info(f'Noise Model: Code Capacity (Perfect Measurement)')
        logging.info(f'Input Structure: Single 2D Image (Depth = 1)')

    logging.info(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    ps_train = args.p_errors
    ps_test = args.p_errors

    # Dataloaders
    # QECC_Dataset 내부에서 args.p_meas를 확인하여 데이터 생성 (3D vs 2D)
    train_loader = DataLoader(
        QECC_Dataset(code, x_error_basis_dict, z_error_basis_dict, ps_train,
                     length=args.samples_per_epoch, args=args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
    )

    # Validation Loaders
    # p_meas > 0 이면 테스트 할 때도 p_meas를 적용해야 함 (보통 p_error와 동일하게 설정하거나 고정값)
    # 여기서는 args.p_meas 값을 그대로 사용
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

    # Training Loop
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
        # test_model 내부에서도 p_meas가 적용된 데이터셋을 사용함
        val_lers = test_model(model, device, val_loaders, ps_test, args.test_samples)
        mean_val_ler = np.mean(val_lers)
        
        # Scheduler step
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
    
    # [추가] Measurement Error (기본값 0.0 -> Code Capacity)
    parser.add_argument('--p_meas', type=float, default=0.0, 
                        help='Measurement error probability. If > 0, inputs are stacked (L+1 times).')

    # Model
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    args = parser.parse_args()

    # Setup output dir
    timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    
    # 폴더 이름에 노이즈 모델 표시
    noise_str = f'phenom_{args.p_meas}' if args.p_meas > 0 else 'code_capacity'
    model_name = f'JungCNN_SGD_{noise_str}'
    
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
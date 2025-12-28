"""
Training Script for 3 Paper Models:
1. Geometry-Aware Transformer (Paper 1)
2. CNN-Transformer / ViT (Paper 2)
3. LUT-Hybrid ViT (Paper 3)
"""
import argparse
import random
import os
import torch
import logging
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# === [핵심] 모델 아키텍처별 클래스 임포트 ===
from qec.models.transformer import ECC_Transformer       # [Paper 1] Geometry-Aware Transformer
from qec.models.vit import ECC_ViT                       # [Paper 2] CNN-Transformer (ViT)
from qec.models.vit import ECC_ViT_LUT_Concat            # [Paper 3] LUT-Hybrid

# 유틸리티 임포트 (기존 코드 재사용)
from qec.core.codes import Get_surface_Code
from qec.training.train_transformer import (
    setup_device, create_surface_code_pure_error_lut, 
    QECC_Dataset, FixedQECC_Dataset, 
    train, test, save_checkpoint, load_checkpoint,
    generate_and_save_dataset, load_dataset, set_seed
)

def main(args):
    # 1. Device Setup
    device = setup_device(args.device)
    
    # 2. Code 객체 완벽 구성 (ECC_Transformer가 내부적으로 좌표 계산할 때 필수)
    class Code: pass
    code = Code()
    
    # Surface Code 행렬 로드
    Hx, Hz, Lx, Lz = Get_surface_Code(args.code_L)
    
    # Tensor 변환 및 Device 할당
    code.H_x = torch.from_numpy(Hx).long().to(device)
    code.H_z = torch.from_numpy(Hz).long().to(device)
    code.L_x = torch.from_numpy(Lx).long().to(device)
    code.L_z = torch.from_numpy(Lz).long().to(device)
    code.pc_matrix = torch.block_diag(code.H_z, code.H_x)
    code.logic_matrix = torch.block_diag(code.L_z, code.L_x)
    code.n = code.pc_matrix.shape[1]
    
    # args에 code 할당 (ECC_Transformer __init__에서 사용됨)
    args.code = code
    args.L = args.code_L 

    # 3. LUT 생성 (Hybrid 모델용 및 데이터 생성용)
    x_error_basis = create_surface_code_pure_error_lut(args.code_L, 'X_only', device)
    z_error_basis = create_surface_code_pure_error_lut(args.code_L, 'Z_only', device)

    # 4. [핵심] 모델 아키텍처 선택 로직
    logging.info(f"Building Model Architecture: {args.model_arch}")

    if args.model_arch == 'transformer':
        # [Paper 1] Geometry-Aware Transformer
        # 특징: Relative Position Bias, 2D Pos Encoding (모델 내부에서 args.code로 좌표 계산)
        model = ECC_Transformer(
            args, 
            dropout=args.dropout, 
            label_smoothing=args.label_smoothing
        )
        
    elif args.model_arch == 'cnn_transformer':
        # [Paper 2] CNN-Transformer (ViT)
        # 특징: 입력 신드롬을 2D 이미지로 변환 후 CNN Patch Embedding
        model = ECC_ViT(
            args, 
            dropout=args.dropout, 
            label_smoothing=args.label_smoothing
        )
        
    elif args.model_arch == 'lut_hybrid':
        # [Paper 3] LUT-Hybrid
        # 특징: 4채널 입력 (Real Z/X + LUT Z/X)
        model = ECC_ViT_LUT_Concat(
            args, 
            x_error_lut=x_error_basis, 
            z_error_lut=z_error_basis, 
            dropout=args.dropout, 
            label_smoothing=args.label_smoothing
        )
    else:
        raise ValueError(f"Unknown architecture: {args.model_arch}")

    model = model.to(device)
    
    # 5. DataParallel (Multi-GPU)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # 6. Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # 7. Dataset & DataLoader 설정
    use_pin_memory = device.type in ['cuda', 'xpu']
    prefetch = 2 if args.workers > 0 else None
    
    # Generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Training Data
    train_dataloader = DataLoader(
        QECC_Dataset(code, x_error_basis, z_error_basis, args.p_errors,
                    len=args.samples_per_epoch, args=args, augment=args.augment),
        batch_size=int(args.batch_size),
        shuffle=True, num_workers=args.workers,
        persistent_workers=True if args.workers > 0 else False,
        pin_memory=use_pin_memory, prefetch_factor=prefetch,
        generator=g
    )

    # Test Data (for each p_error)
    test_dataloader_list = [
        DataLoader(
            QECC_Dataset(code, x_error_basis, z_error_basis, [p],
                        len=int(args.test_batch_size), args=args, seed_offset=10_000_000),
            batch_size=int(args.test_batch_size),
            shuffle=False, num_workers=args.workers,
            persistent_workers=True if args.workers > 0 else False,
            pin_memory=use_pin_memory, prefetch_factor=prefetch
        ) for p in args.p_errors
    ]

    # 8. Training Loop
    logging.info(f"Start Training: {args.epochs} Epochs")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        g.manual_seed(args.seed + epoch)
        
        loss = train(model, device, train_dataloader, optimizer, epoch, LR=scheduler.get_last_lr()[0])
        scheduler.step()

        if loss < best_loss:
            best_loss = loss
            save_checkpoint(model, optimizer, scheduler, epoch, best_loss, args.path)
            logging.info(f'New Best Loss: {best_loss:.5e} (Saved)')

        if epoch % 10 == 0 or epoch == args.epochs:
            test(model, device, test_dataloader_list, args.p_errors, args.test_samples)

    logging.info("Training Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QEC Paper Models Training')
    
    # [새로운 인자] 모델 선택
    parser.add_argument('--model_arch', type=str, default='transformer',
                        choices=['transformer', 'cnn_transformer', 'lut_hybrid'],
                        help='Select model: transformer (Paper1), cnn_transformer (Paper2), lut_hybrid (Paper3)')

    # 기존 인자들
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--code_L', type=int, default=3)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--samples_per_epoch', type=int, default=100000)
    parser.add_argument('--test_samples', type=int, default=10000)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    
    # 노이즈 설정
    parser.add_argument('-p', '--p_errors', type=float, nargs='+',
                        default=[0.07, 0.08, 0.09, 0.1, 0.11, 0.12,0.13])
    parser.add_argument('-y', '--y_ratio', type=float, default=0.0)
    
    # 모델 하이퍼파라미터
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--N_dec', type=int, default=6)
    parser.add_argument('--no_g', type=int, default=0) # Transformer 내부 호환용
    parser.add_argument('--code_type', type=str, default='surface') # 호환용

    args = parser.parse_args()
    
    # 경로 설정
    if args.save_dir:
        args.path = os.path.join(args.save_dir, args.model_arch, f"L{args.code_L}")
        os.makedirs(args.path, exist_ok=True)
        
        # 로깅 설정
        handlers = [logging.FileHandler(os.path.join(args.path, 'train.log')), logging.StreamHandler()]
        logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)

    main(args)
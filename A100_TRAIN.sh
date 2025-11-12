#!/bin/bash
# A100 GPU 최적화 학습 명령어

# L=5 (Standard, 1-2시간)
python -m qec.training.train_with_stim \
    -L 5 \
    --batch_size 2048 \
    --workers 16 \
    --device cuda

# L=7 (Large, 3-4시간)
# python -m qec.training.train_with_stim \
#     -L 7 \
#     --batch_size 1024 \
#     --workers 16 \
#     --N_dec 12 \
#     --device cuda

# L=3 (Quick Test, 20분)
# python -m qec.training.train_with_stim \
#     -L 3 \
#     --batch_size 4096 \
#     --workers 16 \
#     --d_model 128 \
#     --N_dec 6 \
#     --epochs 100 \
#     --patience 20 \
#     --device cuda

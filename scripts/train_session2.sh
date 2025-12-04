#!/bin/bash
# Session 2: 예상 시간 ~64 단위
# 나머지 모델들

set -e
Y_RATIO=0.3333
EPOCHS=50
SAMPLES=2000000

echo "============================================"
echo "Session 2 (예상 시간: ~64 단위)"
echo "============================================"

# ViT_LUT_Concat L7 (16)
echo ">>> L=7 ViT_LUT_Concat"
python -m qec.training.qubit_centric -L 7 -y $Y_RATIO --model_type vit_lut_concat --epochs $EPOCHS --samples_per_epoch $SAMPLES

# Transformer L3,5 (4+8=12)
echo ">>> L=3 Transformer"
python -m qec.training.transformer -L 3 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES
echo ">>> L=5 Transformer"
python -m qec.training.transformer -L 5 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES

# ViT L5 (8)
echo ">>> L=5 ViT"
python -m qec.training.vit -L 5 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES

# Diamond L7 (8)
echo ">>> L=7 Diamond"
python -m qec.training.qubit_centric -L 7 -y $Y_RATIO --model_type diamond --epochs $EPOCHS --samples_per_epoch $SAMPLES

# LUT_Concat L5,7 (4+8=12)
echo ">>> L=5 LUT_Concat"
python -m qec.training.qubit_centric -L 5 -y $Y_RATIO --model_type lut_concat --epochs $EPOCHS --samples_per_epoch $SAMPLES
echo ">>> L=7 LUT_Concat"
python -m qec.training.qubit_centric -L 7 -y $Y_RATIO --model_type lut_concat --epochs $EPOCHS --samples_per_epoch $SAMPLES

# ViT_LUT_Concat L5 (8)
echo ">>> L=5 ViT_LUT_Concat"
python -m qec.training.qubit_centric -L 5 -y $Y_RATIO --model_type vit_lut_concat --epochs $EPOCHS --samples_per_epoch $SAMPLES

echo "============================================"
echo "Session 2 Complete! (8 models)"
echo "============================================"

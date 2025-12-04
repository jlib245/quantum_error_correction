#!/bin/bash
# Session 1: 예상 시간 ~62 단위
# 큰 모델 2개 + 작은 모델들

set -e
Y_RATIO=0.3333
EPOCHS=50
SAMPLES=2000000

echo "============================================"
echo "Session 1 (예상 시간: ~62 단위)"
echo "============================================"

# Transformer L7 (16)
echo ">>> L=7 Transformer"
python -m qec.training.transformer -L 7 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES

# ViT L7 (16)
echo ">>> L=7 ViT"
python -m qec.training.vit -L 7 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES

# FFNN L3,5,7 (1+2+4=7)
echo ">>> L=3 FFNN"
python -m qec.training.ffnn -L 3 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES
echo ">>> L=5 FFNN"
python -m qec.training.ffnn -L 5 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES
echo ">>> L=7 FFNN"
python -m qec.training.ffnn -L 7 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES

# CNN L3,5,7 (1+2+4=7)
echo ">>> L=3 CNN"
python -m qec.training.cnn -L 3 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES
echo ">>> L=5 CNN"
python -m qec.training.cnn -L 5 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES
echo ">>> L=7 CNN"
python -m qec.training.cnn -L 7 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES

# Diamond L3,5 (2+4=6)
echo ">>> L=3 Diamond"
python -m qec.training.qubit_centric -L 3 -y $Y_RATIO --model_type diamond --epochs $EPOCHS --samples_per_epoch $SAMPLES
echo ">>> L=5 Diamond"
python -m qec.training.qubit_centric -L 5 -y $Y_RATIO --model_type diamond --epochs $EPOCHS --samples_per_epoch $SAMPLES

# ViT L3 (4)
echo ">>> L=3 ViT"
python -m qec.training.vit -L 3 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES

# LUT_Concat L3 (2)
echo ">>> L=3 LUT_Concat"
python -m qec.training.qubit_centric -L 3 -y $Y_RATIO --model_type lut_concat --epochs $EPOCHS --samples_per_epoch $SAMPLES

# ViT_LUT_Concat L3 (4)
echo ">>> L=3 ViT_LUT_Concat"
python -m qec.training.qubit_centric -L 3 -y $Y_RATIO --model_type vit_lut_concat --epochs $EPOCHS --samples_per_epoch $SAMPLES

echo "============================================"
echo "Session 1 Complete! (13 models)"
echo "============================================"

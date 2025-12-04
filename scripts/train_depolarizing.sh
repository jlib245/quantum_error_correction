#!/bin/bash
# Batch training script for depolarizing noise (y_ratio = 1/3)
# This trains all decoder models with standard depolarizing noise

set -e  # Exit on error

Y_RATIO=0.3333  # 1/3 for true depolarizing (X=Y=Z)
EPOCHS=50
SAMPLES=2000000

echo "============================================"
echo "Depolarizing Noise Training (y_ratio=$Y_RATIO)"
echo "============================================"

# Function to train with error handling
train_model() {
    local cmd="$1"
    local name="$2"
    echo ""
    echo ">>> Training: $name"
    echo ">>> Command: $cmd"
    eval "$cmd" || echo "WARNING: $name failed, continuing..."
}

# ============================================
# L=3 Training
# ============================================
echo ""
echo "########################################"
echo "# L=3 Training"
echo "########################################"

# FFNN
train_model "python -m qec.training.ffnn -L 3 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L3-FFNN"

# Transformer
train_model "python -m qec.training.transformer -L 3 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L3-Transformer"

# CNN
train_model "python -m qec.training.cnn -L 3 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L3-CNN"

# ViT
train_model "python -m qec.training.vit -L 3 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L3-ViT"

# Diamond CNN
train_model "python -m qec.training.qubit_centric -L 3 -y $Y_RATIO --model_type diamond --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L3-Diamond"

# LUT Concat
train_model "python -m qec.training.qubit_centric -L 3 -y $Y_RATIO --model_type lut_concat --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L3-LUT_Concat"

# ViT LUT Concat
train_model "python -m qec.training.qubit_centric -L 3 -y $Y_RATIO --model_type vit_lut_concat --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L3-ViT_LUT_Concat"

# ============================================
# L=5 Training
# ============================================
echo ""
echo "########################################"
echo "# L=5 Training"
echo "########################################"

# FFNN
train_model "python -m qec.training.ffnn -L 5 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L5-FFNN"

# Transformer
train_model "python -m qec.training.transformer -L 5 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L5-Transformer"

# CNN
train_model "python -m qec.training.cnn -L 5 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L5-CNN"

# ViT
train_model "python -m qec.training.vit -L 5 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L5-ViT"

# Diamond CNN
train_model "python -m qec.training.qubit_centric -L 5 -y $Y_RATIO --model_type diamond --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L5-Diamond"

# LUT Concat
train_model "python -m qec.training.qubit_centric -L 5 -y $Y_RATIO --model_type lut_concat --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L5-LUT_Concat"

# ViT LUT Concat
train_model "python -m qec.training.qubit_centric -L 5 -y $Y_RATIO --model_type vit_lut_concat --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L5-ViT_LUT_Concat"

# ============================================
# L=7 Training
# ============================================
echo ""
echo "########################################"
echo "# L=7 Training"
echo "########################################"

# FFNN
train_model "python -m qec.training.ffnn -L 7 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L7-FFNN"

# Transformer
train_model "python -m qec.training.transformer -L 7 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L7-Transformer"

# CNN
train_model "python -m qec.training.cnn -L 7 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L7-CNN"

# ViT
train_model "python -m qec.training.vit -L 7 -y $Y_RATIO --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L7-ViT"

# Diamond CNN
train_model "python -m qec.training.qubit_centric -L 7 -y $Y_RATIO --model_type diamond --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L7-Diamond"

# LUT Concat
train_model "python -m qec.training.qubit_centric -L 7 -y $Y_RATIO --model_type lut_concat --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L7-LUT_Concat"

# ViT LUT Concat
train_model "python -m qec.training.qubit_centric -L 7 -y $Y_RATIO --model_type vit_lut_concat --epochs $EPOCHS --samples_per_epoch $SAMPLES" "L7-ViT_LUT_Concat"

echo ""
echo "============================================"
echo "All training completed!"
echo "Results saved to Final_Results/surface/L_*/y_$Y_RATIO/"
echo "============================================"

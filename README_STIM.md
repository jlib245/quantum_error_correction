# Stim Training and Evaluation

Simple workflow for training and evaluating QEC decoders with Stim.

## Files

### Training
- `qec/training/train_with_stim.py` - Train Transformer with Stim-generated syndromes

### Evaluation
- `qec/evaluation/compare_decoders_stim.py` - Compare Transformer vs MWPM with Stim

## Usage

### Step 1: Train

```bash
python -m qec.training.train_with_stim \
    -L 5 \
    --epochs 100 \
    --device xpu
```

Output: `Stim_Models/L5_{timestamp}/best_model`

### Step 2: Evaluate

```bash
python -m qec.evaluation.compare_decoders_stim \
    -L 5 \
    --transformer_model Stim_Models/L5_20250112_143022/best_model \
    -p 0.07 0.08 0.09 0.10 0.11 \
    -n 10000
```

Output:
- Log: `experiments_stim/L5_depolarizing/comparison_stim_{timestamp}.log`
- Graph: `experiments_stim/L5_depolarizing/comparison_stim_{timestamp}.png`

## Parameters

### train_with_stim.py
- `-L, --code_L`: Code distance (default: 3)
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Device (cpu/cuda/xpu, default: cuda)
- `--error_rates`: Training error rates (default: 0.07-0.11)

### compare_decoders_stim.py
- `-L`: Code distance
- `--transformer_model`: Path to trained model
- `-p, --p_errors`: Error rates to test
- `-n, --n_shots`: Test shots per error rate (default: 10000)
- `--skip_mwpm`: Skip MWPM evaluation
- `--device`: Device (auto/cpu/cuda/xpu, default: auto)
- `--noise_model`: Stim noise model (depolarizing/SI1000, default: depolarizing)

# Stim Training and Evaluation

Simple workflow for training and evaluating QEC decoders with Stim.

## Files

### Training
- `qec/training/train_with_stim.py` - Train Transformer with Stim-generated syndromes

### Evaluation
- `qec/evaluation/compare_decoders_stim.py` - Compare Transformer vs MWPM with Stim

## Quick Start (GPU 추천)

### Step 1: Train (L=5, Early Stopping)

```bash
python -m qec.training.train_with_stim \
    -L 5 \
    --epochs 100 \
    --batch_size 256 \
    --patience 20 \
    --device xpu
```

Output: `Stim_Models/L5_{timestamp}/best_model`

### Step 2: Evaluate (50k shots)

```bash
python -m qec.evaluation.compare_decoders_stim \
    -L 5 \
    --transformer_model Stim_Models/L5_20250112_143022/best_model \
    -p 0.07 0.08 0.09 0.10 0.11 \
    -n 50000 \
    --device xpu
```

Output:
- Log: `experiments_stim/L5_depolarizing/comparison_stim_{timestamp}.log`
- Graph: `experiments_stim/L5_depolarizing/comparison_stim_{timestamp}.png`

---

## Parameters

### train_with_stim.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-L, --code_L` | 3 | Code distance |
| `--epochs` | 100 | Maximum training epochs |
| `--batch_size` | 128 | Batch size (GPU: 256 추천) |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 20 | Early stopping patience (0=disabled) |
| `--min_delta` | 0.0 | Minimum loss improvement |
| `--workers` | 4 | Data loading workers (8 추천) |
| `--device` | cuda | Device (cpu/cuda/xpu) |
| `--d_model` | 128 | Transformer dimension |
| `--h` | 16 | Attention heads |
| `--N_dec` | 6 | Decoder layers |
| `--error_rates` | 0.07-0.11 | Training error rates |

### compare_decoders_stim.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-L` | - | Code distance |
| `--transformer_model` | - | Path to trained model |
| `-p, --p_errors` | - | Error rates to test |
| `-n, --n_shots` | 10000 | Test shots per error rate |
| `--skip_mwpm` | False | Skip MWPM evaluation |
| `--device` | auto | Device (auto/cpu/cuda/xpu) |
| `--noise_model` | depolarizing | Stim noise model |
| `--rounds` | 1 | QEC rounds |

---

## Distance별 추천 설정 (GPU)

### L=3 (Quick Test)
```bash
# Train (~10분)
python -m qec.training.train_with_stim \
    -L 3 --epochs 50 --batch_size 512 --patience 15 --device xpu

# Eval (~3분)
python -m qec.evaluation.compare_decoders_stim \
    -L 3 --transformer_model <model> -p 0.07 0.08 0.09 -n 30000 --device xpu
```

### L=5 (Standard)
```bash
# Train (~30분)
python -m qec.training.train_with_stim \
    -L 5 --epochs 100 --batch_size 256 --patience 20 --device xpu

# Eval (~8분)
python -m qec.evaluation.compare_decoders_stim \
    -L 5 --transformer_model <model> -p 0.07 0.08 0.09 0.10 0.11 -n 50000 --device xpu
```

### L=7 (Research)
```bash
# Train (~2시간)
python -m qec.training.train_with_stim \
    -L 7 --epochs 150 --batch_size 128 --patience 25 --lr 0.0005 --device xpu

# Eval (~30분)
python -m qec.evaluation.compare_decoders_stim \
    -L 7 --transformer_model <model> -p 0.08 0.09 0.10 0.11 0.12 -n 100000 --device xpu
```

---

## Early Stopping 설명

- `--patience 20`: 20 epoch 동안 loss 개선 없으면 중단
- `--min_delta 0.0`: 개선 최소 임계값 (보통 0.0 사용)

**예시**:
```
Epoch 45: Loss=0.2341, Acc=0.9123
  → Saved best model (loss=0.2341)
Epoch 46: Loss=0.2345, Acc=0.9119
  No improvement. Patience: 1/20
...
Epoch 65: Loss=0.2343, Acc=0.9120
  No improvement. Patience: 20/20

Early stopping triggered after 65 epochs (patience=20)
Training complete! Model saved to: Stim_Models/L5_xxx
Best loss: 0.2341
```

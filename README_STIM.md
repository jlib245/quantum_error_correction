# Stim Training and Evaluation

Simple workflow for training and evaluating QEC decoders with Stim.

## Files

### Training
- `qec/training/train_with_stim.py` - Train Transformer with Stim-generated syndromes

### Evaluation
- `qec/evaluation/compare_decoders_stim.py` - Compare Transformer vs MWPM with Stim

## Quick Start (GPU 필수)

### Step 1: Train (L=5, 기본 설정)

```bash
# 기본 설정 (대형 모델: d_model=256, N_dec=10, batch=1024)
python -m qec.training.train_with_stim --device xpu
```

Output: `Stim_Models/L5_{timestamp}/best_model`

**기본 설정**:
- L=5, epochs=300 (early stopping patience=40)
- d_model=256, N_dec=10, h=16
- batch_size=1024, workers=8
- 예상 시간: 1-2시간 (A100 기준)

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
| `-L, --code_L` | 5 | Code distance |
| `--epochs` | 300 | Maximum training epochs |
| `--batch_size` | 1024 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 40 | Early stopping patience (0=disabled) |
| `--min_delta` | 0.0 | Minimum loss improvement |
| `--workers` | 8 | Data loading workers |
| `--device` | cuda | Device (cpu/cuda/xpu) |
| `--d_model` | 256 | Transformer dimension |
| `--h` | 16 | Attention heads |
| `--N_dec` | 10 | Decoder layers |
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

### L=3 (Quick Test - 작은 모델)
```bash
# Train (~20분, 작은 모델)
python -m qec.training.train_with_stim \
    -L 3 \
    --d_model 128 \
    --N_dec 6 \
    --batch_size 512 \
    --epochs 100 \
    --patience 20 \
    --device xpu

# Eval (~3분)
python -m qec.evaluation.compare_decoders_stim \
    -L 3 --transformer_model <model> -p 0.07 0.08 0.09 -n 30000 --device xpu
```

### L=5 (Standard - 기본 설정)
```bash
# Train (~1-2시간, 대형 모델)
python -m qec.training.train_with_stim --device xpu
# 또는 명시적으로:
# python -m qec.training.train_with_stim \
#     -L 5 --d_model 256 --N_dec 10 --batch_size 1024 --device xpu

# Eval (~10분)
python -m qec.evaluation.compare_decoders_stim \
    -L 5 --transformer_model <model> -p 0.07 0.08 0.09 0.10 0.11 -n 50000 --device xpu
```

### L=7 (Research - 초대형 모델)
```bash
# Train (~3-4시간, 초대형 모델)
python -m qec.training.train_with_stim \
    -L 7 \
    --d_model 256 \
    --N_dec 12 \
    --batch_size 512 \
    --epochs 300 \
    --patience 50 \
    --device xpu

# Eval (~40분)
python -m qec.evaluation.compare_decoders_stim \
    -L 7 --transformer_model <model> -p 0.08 0.09 0.10 0.11 0.12 -n 100000 --device xpu
```

---

## Early Stopping 설명

- `--patience 40` (기본값): 40 epoch 동안 loss 개선 없으면 중단
- `--min_delta 0.0`: 개선 최소 임계값 (보통 0.0 사용)
- 기존 `train_transformer.py`와 동일한 설정

**예시**:
```
Epoch 125: Loss=0.2341, Acc=0.9123
  → Saved best model (loss=0.2341)
Epoch 126: Loss=0.2345, Acc=0.9119
  No improvement. Patience: 1/40
...
Epoch 165: Loss=0.2343, Acc=0.9120
  No improvement. Patience: 40/40

Early stopping triggered after 165 epochs (patience=40)
Training complete! Model saved to: Stim_Models/L5_xxx
Best loss: 0.2341
```

**참고**: 기존 모델 로그와 동일한 patience=40 사용

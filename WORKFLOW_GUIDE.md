# Complete Workflow Guide

양자 오류 정정 Transformer 모델의 **학습부터 현실적 평가까지** 전체 워크플로우 가이드입니다.

---

## 📋 전체 파이프라인

```
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: Training (train_transformer.py)                      │
│ - Transformer 모델 학습                                        │
│ - Surface Code 시뮬레이션 데이터                               │
│ - 100k samples/epoch, Early stopping                         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: Quick Evaluation (compare_decoders.py)               │
│ - 수학적 시뮬레이션 (빠름)                                      │
│ - Transformer vs MWPM 비교                                    │
│ - 다양한 오류율 테스트                                          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: Realistic Simulation (Stim + Transformer)            │
│ - Stim으로 현실적 양자 오류 생성                                │
│ - Google Quantum AI 수준 시뮬레이션                            │
│ - 최종 성능 평가                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 방법 1: 자동 워크플로우 (권장)

### 전체 파이프라인 한 번에 실행

```bash
python run_full_workflow.py \
    --code_L 3 \
    --epochs 100 \
    --batch_size 128 \
    --error_rates 0.07 0.08 0.09 0.10 0.11 \
    --quick_shots 5000 \
    --realistic_shots 10000 \
    --device auto
```

**예상 시간**:
- 학습: ~30분 (GPU) / ~2시간 (CPU)
- Quick Evaluation: ~5분
- Realistic Simulation: ~10분
- **총: ~45분 (GPU) / ~2.5시간 (CPU)**

---

### 기존 모델로 평가만 실행

```bash
python run_full_workflow.py \
    --skip_training \
    --model_path Final_Results_QECCT/surface/.../best_model \
    --error_rates 0.07 0.09 0.11 \
    --realistic_shots 10000
```

---

## 🔧 방법 2: 단계별 실행 (세밀한 제어)

### STEP 1: 모델 학습

```bash
cd qec/training
python train_transformer.py \
    --code_L 3 \
    --epochs 100 \
    --batch_size 128 \
    --y_ratio 0.0 \
    --patience 20 \
    --device auto
```

**결과물**:
```
Final_Results_QECCT/
└── surface/
    └── Transformer_Code_L_3/
        └── noise_model_independent/
            └── repetition_1/
                └── 12_11_2025_18_30_45/
                    ├── best_model          ← 이것 사용
                    ├── final_model
                    └── logging.txt
```

---

### STEP 2: 빠른 평가 (수학적 시뮬레이션)

```bash
cd qec/evaluation
python compare_decoders.py \
    --model_transformer ../../Final_Results_QECCT/surface/.../best_model \
    --code_L 3 \
    --p_range 0.07 0.08 0.09 0.10 0.11 \
    --n_shots 5000 \
    --device auto
```

**출력 예시**:
```
============================================================
TRANSFORMER Model Evaluation
============================================================

Testing p=0.070...
  LER: 2.340e-02
  Avg Latency: 0.523 ms
  Logical Errors: 117/5000

Testing p=0.090...
  LER: 5.120e-02
  ...
```

---

### STEP 3: 현실적 시뮬레이션 (Stim)

#### Python 스크립트로 실행

```python
from qec.evaluation.realistic_simulation import run_realistic_evaluation

results = run_realistic_evaluation(
    model_path='Final_Results_QECCT/surface/.../best_model',
    distance=3,
    error_rates=[0.07, 0.08, 0.09, 0.10, 0.11],
    shots=10000,
    device='cpu',
    verbose=True
)

# 결과 확인
for p, result in results.items():
    print(f"p={p:.3f}: LER={result.logical_error_rate:.6f}, "
          f"Accuracy={result.decoder_accuracy:.4f}")
```

#### 커맨드라인으로 실행

```bash
cd examples
python run_quantum_simulation.py \
    --model_path ../Final_Results_QECCT/surface/.../best_model \
    --distance 3 \
    --error_rate 0.09 \
    --shots 10000 \
    --device cpu
```

---

## 📊 결과 해석

### Quick Evaluation vs Realistic Simulation

| 항목 | Quick Eval (compare_decoders) | Realistic (Stim) |
|------|------------------------------|------------------|
| **오류 생성** | NumPy random | Stim (현실적) |
| **속도** | 빠름 (~5분) | 중간 (~10분) |
| **정확도** | 근사치 | 실제와 유사 |
| **용도** | 개발/디버깅 | 최종 평가 |

**예시**:
```
Quick Eval  (p=0.09): LER=0.0512, Accuracy=91.2%
Realistic   (p=0.09): LER=0.0523, Accuracy=89.8%
Difference: LER +0.0011 (2.1% 차이)
```

→ **차이가 작음**: Transformer가 현실적 노이즈에도 잘 작동

---

## 🧪 고급 사용법

### 1. Y 오류 비율 실험

```bash
# Y 오류 30%로 학습
python run_full_workflow.py \
    --code_L 3 \
    --y_ratio 0.3 \
    --epochs 100

# 결과: Y 오류 환경에 특화된 모델
```

---

### 2. 다양한 거리 비교

```bash
# L=3, 5, 7 각각 학습 및 평가
for L in 3 5 7; do
    python run_full_workflow.py \
        --code_L $L \
        --epochs 100 \
        --realistic_shots 10000
done
```

**기대 결과**:
- L 증가 → LER 감소 (더 강력한 보호)
- L 증가 → 학습 시간 증가

---

### 3. MWPM과 Transformer 비교

```bash
cd qec/evaluation
python compare_decoders.py \
    --model_transformer <transformer_model> \
    --code_L 3 \
    --p_range 0.07 0.09 0.11 \
    --n_shots 10000 \
    --compare_mwpm
```

**출력**:
```
Decoder      | p=0.07 LER | p=0.09 LER | p=0.11 LER
-----------------------------------------------------------
MWPM         | 2.1e-02    | 4.8e-02    | 9.2e-02
Transformer  | 2.3e-02    | 5.1e-02    | 8.9e-02
```

---

### 4. 수백만 샷 대규모 시뮬레이션

```python
from qec.evaluation.realistic_simulation import run_realistic_evaluation

# 100만 샷 (통계적으로 매우 정확)
results = run_realistic_evaluation(
    model_path='path/to/model',
    distance=3,
    error_rates=[0.09],
    shots=1000000,  # ← 1M shots
    device='cuda',   # GPU 필수
    verbose=True
)

print(f"LER with 1M shots: {results[0.09].logical_error_rate:.8f}")
# 예: LER with 1M shots: 0.05234821
```

---

## 📈 벤치마크 예시

### L=3, p=0.09 기준 (Intel i7 + NVIDIA RTX 3060)

| 단계 | 시간 | 메모리 |
|------|------|--------|
| Training (100 epochs) | 28분 | 2.1 GB |
| Quick Eval (5k shots) | 4.2분 | 1.2 GB |
| Realistic Sim (10k shots) | 8.7분 | 1.5 GB |
| **Total** | **41분** | **2.1 GB** |

---

## 🐛 문제 해결

### 1. 모델 로드 오류

```
Error: Model file not found
```

**해결**:
```bash
# 최신 모델 찾기
ls -lt Final_Results_QECCT/surface/Transformer_Code_L_3/*/*/* | head -1

# 정확한 경로 사용
python run_full_workflow.py \
    --skip_training \
    --model_path <정확한_경로>/best_model
```

---

### 2. Stim 설치 오류

```
ModuleNotFoundError: No module named 'stim'
```

**해결**:
```bash
pip install stim
```

---

### 3. GPU 메모리 부족

```
CUDA out of memory
```

**해결**:
```bash
# 배치 크기 줄이기
python run_full_workflow.py \
    --batch_size 64 \
    --realistic_shots 5000
```

---

### 4. Syndrome dimension mismatch

```
RuntimeError: size mismatch
```

**해결**:
- 모델 학습 시 사용한 `code_L`과 동일한 값 사용
- L=3 모델 → distance=3으로 평가
- L=5 모델 → distance=5로 평가

---

## 📚 디렉토리 구조

```
quantum_error_correction/
├── run_full_workflow.py        ← 전체 워크플로우
├── qec/
│   ├── training/
│   │   ├── train_transformer.py
│   │   └── train_ffnn.py
│   ├── evaluation/             ← 새로 추가!
│   │   ├── compare_decoders.py
│   │   └── realistic_simulation.py
│   ├── simulation/             ← 새로 추가!
│   │   ├── quantum_simulator.py
│   │   └── __init__.py
│   ├── models/
│   ├── decoders/
│   └── core/
├── examples/
│   ├── run_quantum_simulation.py
│   └── demo_quantum_computer.py
└── Final_Results_QECCT/        ← 학습 결과
    └── surface/
        └── Transformer_Code_L_3/
```

---

## 💡 베스트 프랙티스

### 1. 개발 단계
```bash
# 빠른 반복: Quick Eval만 사용
python run_full_workflow.py \
    --epochs 50 \
    --skip_realistic \
    --quick_shots 1000
```

### 2. 최종 평가 단계
```bash
# 현실적 평가: Stim으로 충분한 샷 수
python run_full_workflow.py \
    --skip_training \
    --skip_quick_eval \
    --realistic_shots 50000
```

### 3. 논문 작성 단계
```bash
# 대규모 시뮬레이션
python run_full_workflow.py \
    --skip_training \
    --realistic_shots 1000000 \
    --device cuda
```

---

## 🎓 요약

1. **학습**: `train_transformer.py` (또는 `run_full_workflow.py`)
2. **빠른 평가**: `compare_decoders.py` (수학적 시뮬레이션)
3. **현실적 평가**: `run_quantum_simulation.py` (Stim)
4. **전체 자동화**: `run_full_workflow.py --device auto`

**추천 워크플로우**:
```bash
# 한 줄로 전체 실행
python run_full_workflow.py --code_L 3 --epochs 100 --device auto
```

이제 실제 양자 컴퓨터에서 돌릴 수 있는 수준의 시뮬레이션을 수행할 수 있습니다! 🚀

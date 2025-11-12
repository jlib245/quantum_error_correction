# Quantum Computer Simulation Guide

이 가이드는 Stim을 사용한 양자 컴퓨터 시뮬레이션 사용법을 설명합니다.

## 🎯 개요

우리의 시뮬레이터는 다음을 수행합니다:

1. **Stim** (Google Quantum AI): 실제 양자 오류를 현실적으로 시뮬레이션
2. **Transformer Decoder**: 학습된 모델로 신드롬을 디코딩
3. **통계 분석**: 논리 오류율, 디코더 정확도 등 계산

```
┌─────────────────────────────────────────────────┐
│ 실제 양자 컴퓨터 (Google/IBM/...)                │
│ - 물리 큐빗에 오류 발생                           │
│ - Stabilizer 측정 → 신드롬                       │
│ - 디코더 실행 → 오류 보정                         │
└─────────────────────────────────────────────────┘
                    ↓ 시뮬레이션
┌─────────────────────────────────────────────────┐
│ 우리의 시뮬레이터 (Stim + Transformer)            │
│ - Stim: 현실적 오류 생성                         │
│ - Transformer: 신드롬 디코딩                     │
│ - 통계: LER, 정확도 측정                         │
└─────────────────────────────────────────────────┘
```

---

## 📦 설치

```bash
# Stim 설치 (이미 완료됨)
pip install stim

# 기존 의존성 확인
pip install torch numpy
```

---

## 🚀 빠른 시작

### 1. 기본 데모 실행 (모델 없이)

```bash
cd examples
python demo_quantum_computer.py --demo stim
```

**출력 예시**:
```
============================================================
DEMO 1: Stim Basics
============================================================

Circuit Info:
  Distance: 3
  Physical qubits: 9
  Detectors (syndromes): 8
  Observables (logical qubits): 1

Sample Syndromes (first 5):
  Shot 1: syndrome=00110010, logical_error=0
  Shot 2: syndrome=10000100, logical_error=1
  Shot 3: syndrome=00000000, logical_error=0
  Shot 4: syndrome=01010000, logical_error=0
  Shot 5: syndrome=11001001, logical_error=1
```

---

### 2. 신드롬 통계 확인

```bash
python demo_quantum_computer.py --demo stats
```

**출력 예시**:
```
Sampling 1000 shots at each error rate:
 Error Rate | Syndromes!=0 | Logical Errors | Avg Syndrome Weight
---------------------------------------------------------------------------
       0.050 |          315 |             12 |                 0.82
       0.070 |          432 |             28 |                 1.15
       0.090 |          548 |             51 |                 1.52
       0.110 |          641 |             89 |                 1.91
       0.130 |          712 |            134 |                 2.34
```

**해석**:
- 오류율 증가 → 신드롬 발생 증가
- 오류율 증가 → 논리 오류 증가 (threshold ~11% 근처)
- 평균 신드롬 가중치 증가 (더 많은 stabilizer 위반)

---

### 3. Transformer 디코더 테스트

먼저 모델을 학습해야 합니다:

```bash
cd ../qec/training
python train_transformer.py --code_L 3 --epochs 50 --batch_size 128
```

학습 완료 후:

```bash
cd ../../examples
python demo_quantum_computer.py \
    --demo decoder \
    --model_path ../Final_Results_QECCT/surface/Transformer_Code_L_3/.../best_model
```

**출력 예시**:
```
Shot | Syndrome                 |   Actual |  Predicted | Match
-----------------------------------------------------------------
   1 | 00000000                 |        I |          I |      ✓
   2 | 00110010                 |        I |          I |      ✓
   3 | 10000101                 |        X |          X |      ✓
   4 | 01010000                 |        I |          X |      ✗
   5 | 11001100                 |        X |          X |      ✓
  ...

Decoder Accuracy: 87.50%
```

---

### 4. 전체 양자 컴퓨터 시뮬레이션

```bash
python run_quantum_simulation.py \
    --model_path <모델_경로> \
    --distance 3 \
    --error_rate 0.09 \
    --shots 10000
```

**출력 예시**:
```
============================================================
Running Quantum Computer Simulation
============================================================
Distance: 3
Physical error rate: 0.09
Shots: 10000
============================================================

[1/4] Generating realistic syndrome samples with Stim...
  ✓ Generated 10000 samples
  ✓ Syndrome dimension: 8
  ✓ Logical errors occurred: 523 / 10000 (5.23%)

[2/4] Formatting syndromes for Transformer...

[3/4] Decoding syndromes with Transformer...
  ✓ Decoded 10000 syndromes
  ✓ Predictions: I=9102 X=898

[4/4] Calculating statistics...

============================================================
Simulation Results
============================================================
Total shots: 10000
Logical errors: 523 (5.230%)
Decoder accuracy: 91.23%
Execution time: 2.45s
============================================================
```

---

## 📊 주요 지표 해석

### 1. **Logical Error Rate (LER)**
```python
LER = (논리 오류 발생 횟수) / (총 샷 수)
```

**의미**:
- **낮을수록 좋음** (Surface Code가 제대로 작동)
- p < threshold: LER 감소
- p > threshold: LER 증가

**예시**:
```
p=0.07: LER=2.8%  ← 코드 효과적
p=0.09: LER=5.2%
p=0.11: LER=8.9%  ← threshold 근처
p=0.13: LER=13.4% ← 코드 무력화
```

---

### 2. **Decoder Accuracy**
```python
Accuracy = (올바른 예측 횟수) / (총 샷 수)
```

**의미**:
- Transformer가 신드롬 → 논리 오류를 얼마나 정확히 예측하는가
- **높을수록 좋음** (디코더 성능 우수)

**예시**:
```
Decoder Accuracy: 91.23%
→ 10000번 중 9123번 올바른 예측
→ 877번 잘못 예측 (하지만 일부는 무해할 수 있음)
```

---

### 3. **Syndrome Weight**
```python
Weight = (켜진 신드롬 비트 수)
```

**의미**:
- 평균 신드롬 가중치 ≈ 오류 개수
- 높을수록 오류 많음

---

## 🔬 고급 사용법

### 다양한 오류율로 실험

```bash
for p in 0.07 0.08 0.09 0.10 0.11; do
    python run_quantum_simulation.py \
        --model_path <모델_경로> \
        --distance 3 \
        --error_rate $p \
        --shots 5000
done
```

---

### GPU 사용 (빠른 디코딩)

```bash
python run_quantum_simulation.py \
    --model_path <모델_경로> \
    --distance 3 \
    --error_rate 0.09 \
    --shots 100000 \
    --device cuda
```

---

### 다양한 거리 비교

```bash
# L=3
python run_quantum_simulation.py --model_path model_L3 --distance 3 --shots 10000

# L=5
python run_quantum_simulation.py --model_path model_L5 --distance 5 --shots 10000

# L=7
python run_quantum_simulation.py --model_path model_L7 --distance 7 --shots 10000
```

**기대 결과**:
- 거리 증가 → LER 감소 (더 강력한 보호)
- 거리 증가 → 계산 시간 증가

---

## 📈 결과 분석 예시

### Threshold 찾기

여러 오류율에서 시뮬레이션을 실행하고 LER을 플롯:

```python
import matplotlib.pyplot as plt

error_rates = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
lers = []

for p in error_rates:
    result = run_simulation(model_path, distance=3, physical_error_rate=p, shots=10000)
    lers.append(result.logical_error_rate)

plt.plot(error_rates, lers, 'o-')
plt.xlabel('Physical Error Rate')
plt.ylabel('Logical Error Rate')
plt.title('Surface Code Performance (L=3)')
plt.axvline(0.11, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()
```

---

## 🧪 실험 아이디어

### 1. **Transformer vs MWPM 비교**
```bash
# Transformer
python run_quantum_simulation.py --model_path transformer_model ...

# MWPM (기존 compare_decoders.py 사용)
python ../qec/training/compare_decoders.py --decoder mwpm ...
```

### 2. **Y 오류 비율 실험**
```bash
# 학습: Y 오류 30%
python train_transformer.py --y_ratio 0.3

# 테스트: Y 오류 0% (independent)
python run_quantum_simulation.py --model_path model_y0.3 ...

# 일반화 성능 확인
```

### 3. **대규모 시뮬레이션**
```bash
# 100만 샷 (통계적으로 정확)
python run_quantum_simulation.py \
    --model_path <모델> \
    --shots 1000000 \
    --device cuda
```

---

## 🐛 문제 해결

### 오류: "Module 'stim' not found"
```bash
pip install stim
```

### 오류: "Model file not found"
```bash
# 모델 경로 확인
ls Final_Results_QECCT/surface/Transformer_Code_L_3/

# 정확한 타임스탬프 폴더 사용
python run_quantum_simulation.py \
    --model_path Final_Results_QECCT/surface/Transformer_Code_L_3/noise_model_independent/repetition_1/DD_MM_YYYY_HH_MM_SS/best_model
```

### 오류: Syndrome dimension mismatch
```bash
# 모델 학습 시 사용한 L과 동일한 L 사용
# L=3 모델 → distance=3
# L=5 모델 → distance=5
```

---

## 📚 추가 자료

- **Stim 문서**: https://github.com/quantumlib/Stim
- **Surface Code 튜토리얼**: https://arxiv.org/abs/1208.0928
- **Transformer 학습 가이드**: `../docs/training_process_detailed.md`

---

## 💡 핵심 포인트

1. **Stim은 현실적 양자 오류를 시뮬레이션** (Google에서 개발)
2. **Transformer는 신드롬만 보고 논리 오류 예측** (학습 기반)
3. **LER이 낮을수록 양자 컴퓨터가 안정적**
4. **Decoder Accuracy가 높을수록 디코더 성능 우수**

이제 실제 양자 컴퓨터처럼 시뮬레이션을 돌려보세요! 🚀

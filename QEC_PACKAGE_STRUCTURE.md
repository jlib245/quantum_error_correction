# QEC Package Structure

양자 오류 정정 (Quantum Error Correction) 패키지 구조 및 사용법

---

## 📦 패키지 구조

```
quantum_error_correction/
│
├── run_full_workflow.py          ← 전체 워크플로우 실행
├── WORKFLOW_GUIDE.md              ← 사용 가이드
│
├── qec/                           ← 메인 패키지
│   │
│   ├── core/                      ← 핵심 코드 유틸리티
│   │   ├── codes.py               - Surface/Toric Code 생성
│   │   └── __init__.py
│   │
│   ├── models/                    ← 신경망 모델
│   │   ├── transformer.py         - Transformer 디코더
│   │   ├── ffnn.py                - Feed-Forward NN 디코더
│   │   └── __init__.py
│   │
│   ├── decoders/                  ← 전통적 디코더
│   │   ├── mwpm.py                - Minimum Weight Perfect Matching
│   │   └── __init__.py
│   │
│   ├── training/                  ← 모델 학습
│   │   ├── train_transformer.py   - Transformer 학습 스크립트
│   │   ├── train_ffnn.py          - FFNN 학습 스크립트
│   │   ├── test_mwpm.py           - MWPM 테스트
│   │   └── __init__.py
│   │
│   ├── evaluation/                ← 평가 모듈 (새로 추가!)
│   │   ├── compare_decoders.py    - 빠른 평가 (수학적 시뮬레이션)
│   │   ├── realistic_simulation.py - 현실적 평가 (Stim)
│   │   └── __init__.py
│   │
│   ├── simulation/                ← 양자 시뮬레이션 (새로 추가!)
│   │   ├── quantum_simulator.py   - Stim 기반 양자 컴퓨터 시뮬레이터
│   │   └── __init__.py
│   │
│   ├── data/                      ← 데이터
│   │   └── codes_db/              - 사전 계산된 코드 행렬
│   │
│   └── utils/                     ← 유틸리티
│       └── __init__.py
│
├── examples/                      ← 사용 예시
│   ├── run_quantum_simulation.py  - Stim 시뮬레이션 실행
│   ├── demo_quantum_computer.py   - 데모 스크립트
│   └── README_SIMULATION.md       - 시뮬레이션 가이드
│
├── docs/                          ← 문서
│   └── training_process_detailed.md
│
└── Final_Results_QECCT/           ← 학습 결과 (자동 생성)
    └── surface/
        └── Transformer_Code_L_*/
```

---

## 🔄 워크플로우

### 방법 1: 자동 실행 (권장)

```python
# 루트 디렉토리에서
python run_full_workflow.py --code_L 3 --epochs 100
```

**내부 동작**:
1. `qec.training.train_transformer` → 모델 학습
2. `qec.evaluation.compare_decoders` → 빠른 평가
3. `qec.evaluation.realistic_simulation` → Stim 평가

---

### 방법 2: 모듈별 사용

#### 1. 학습

```python
from qec.training.train_transformer import main as train_transformer
import argparse

args = argparse.Namespace(
    code_L=3,
    epochs=100,
    batch_size=128,
    device='auto',
    # ... 기타 설정
)

train_transformer(args)
```

#### 2. 빠른 평가

```python
from qec.evaluation.compare_decoders import evaluate_nn_model
from qec.core.codes import Get_surface_Code

Hx, Hz, Lx, Lz = Get_surface_Code(L=3)

results = evaluate_nn_model(
    model_path='path/to/best_model',
    model_type='transformer',
    Hx=Hx, Hz=Hz, Lx=Lx, Lz=Lz,
    p_errors=[0.07, 0.09, 0.11],
    n_shots=5000
)
```

#### 3. 현실적 시뮬레이션

```python
from qec.evaluation.realistic_simulation import run_realistic_evaluation

results = run_realistic_evaluation(
    model_path='path/to/best_model',
    distance=3,
    error_rates=[0.07, 0.09, 0.11],
    shots=10000,
    device='cpu'
)

for p, result in results.items():
    print(f"p={p}: LER={result.logical_error_rate:.6f}")
```

#### 4. 직접 양자 시뮬레이터 사용

```python
from qec.simulation import QuantumComputer, TransformerDecoder

# 디코더 로드
decoder = TransformerDecoder('path/to/best_model', device='cpu')

# 양자 컴퓨터 생성
qc = QuantumComputer(
    distance=3,
    decoder=decoder,
    physical_error_rate=0.09
)

# 시뮬레이션 실행
result = qc.run_simulation(shots=10000, verbose=True)

print(f"Logical Error Rate: {result.logical_error_rate:.6f}")
print(f"Decoder Accuracy: {result.decoder_accuracy:.4f}")
```

---

## 📚 주요 클래스 및 함수

### `qec.core.codes`

```python
Get_surface_Code(L: int) -> (Hx, Hz, Lx, Lz)
```
Surface Code 행렬 로드

---

### `qec.models.transformer`

```python
class ECC_Transformer(nn.Module):
    """Transformer 기반 디코더"""
    def __init__(self, args, dropout=0)
    def forward(self, syndrome) -> logits  # (batch, 4) - I/X/Z/Y
    def loss(self, pred, true_label) -> loss
```

---

### `qec.decoders.mwpm`

```python
class MWPM_Decoder:
    """Minimum Weight Perfect Matching 디코더"""
    def __init__(self, Hx, Hz, Lx, Lz)
    def decode(self, syndrome) -> predicted_class
    def evaluate(self, p, n_shots) -> results
```

---

### `qec.training.train_transformer`

```python
# 데이터셋
class QECC_Dataset(data.Dataset):
    def __getitem__(self, index) -> (syndrome, true_class)

# LUT 생성
create_surface_code_pure_error_lut(L, error_type, device) -> lut_dict

# Simple Decoder C
simple_decoder_C_torch(syndrome, x_lut, z_lut, H_z, H_x) -> pure_error
```

---

### `qec.simulation.quantum_simulator`

```python
class StimSurfaceCodeSimulator:
    """Stim 기반 Surface Code 시뮬레이터"""
    def __init__(self, distance, rounds, physical_error_rate, ...)
    def sample_syndromes(self, shots) -> (syndromes, logical_errors)

class TransformerDecoder:
    """Transformer 디코더 래퍼"""
    def __init__(self, model_path, device)
    def decode_batch(self, syndromes) -> predictions

class QuantumComputer:
    """완전한 양자 컴퓨터 시뮬레이션"""
    def __init__(self, distance, decoder, physical_error_rate, ...)
    def run_simulation(self, shots, verbose) -> SimulationResult

@dataclass
class SimulationResult:
    total_shots: int
    logical_errors: int
    decoder_predictions: np.ndarray
    decoder_accuracy: float
    logical_error_rate: float
    execution_time: float
```

---

### `qec.evaluation.realistic_simulation`

```python
run_realistic_evaluation(
    model_path: str,
    distance: int,
    error_rates: List[float],
    shots: int,
    device: str
) -> Dict[float, SimulationResult]
```

---

## 🎯 사용 시나리오별 가이드

### 시나리오 1: 빠른 프로토타입

```bash
# 학습 (간단)
python qec/training/train_transformer.py --code_L 3 --epochs 50

# 평가 (빠름)
python qec/evaluation/compare_decoders.py \
    --model_transformer <모델> \
    --code_L 3 \
    --n_shots 1000
```

---

### 시나리오 2: 논문 작성

```bash
# 전체 워크플로우 (현실적 평가 포함)
python run_full_workflow.py \
    --code_L 3 \
    --epochs 200 \
    --realistic_shots 100000 \
    --device cuda
```

---

### 시나리오 3: 커스텀 실험

```python
# Python 스크립트로 세밀한 제어
from qec.simulation import QuantumComputer, TransformerDecoder
from qec.simulation.quantum_simulator import StimSurfaceCodeSimulator

# 1. 커스텀 Stim 회로
simulator = StimSurfaceCodeSimulator(
    distance=5,
    rounds=10,  # 시간 상관 오류
    noise_model='SI1000',  # Google Sycamore 노이즈
    physical_error_rate=0.001,
    measurement_error_rate=0.0001
)

# 2. 신드롬 샘플링
syndromes, errors = simulator.sample_syndromes(shots=1000)

# 3. 디코더 예측
decoder = TransformerDecoder('path/to/model')
predictions = decoder.decode_batch(syndromes)

# 4. 분석
accuracy = (predictions == errors.flatten()).sum() / len(errors)
print(f"Accuracy: {accuracy:.4f}")
```

---

## 🔧 확장 가능성

### 새로운 디코더 추가

```python
# qec/decoders/my_decoder.py
class MyDecoder:
    def __init__(self, Hx, Hz, Lx, Lz):
        self.Hx = Hx
        # ...

    def decode(self, syndrome):
        # 커스텀 디코딩 로직
        return predicted_class

    def evaluate(self, p, n_shots):
        # 평가 로직
        return results
```

### 새로운 노이즈 모델 추가

```python
# qec/simulation/quantum_simulator.py
class StimSurfaceCodeSimulator:
    def _create_circuit(self):
        if self.noise_model == 'my_custom_noise':
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                # 커스텀 노이즈 파라미터
                ...
            )
        return circuit
```

---

## 📊 디버깅 팁

### 1. 모델 출력 확인

```python
from qec.simulation import TransformerDecoder
import torch

decoder = TransformerDecoder('path/to/model')

# 단일 신드롬 테스트
syndrome = torch.tensor([[0,0,1,1,0,0,0,0]]).float()
with torch.no_grad():
    logits = decoder.model(syndrome)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(logits, dim=1)

print(f"Logits: {logits}")
print(f"Probabilities (I/X/Z/Y): {probs}")
print(f"Prediction: {pred.item()}")
```

### 2. LUT 검증

```python
from qec.training.train_transformer import create_surface_code_pure_error_lut

lut = create_surface_code_pure_error_lut(L=3, error_type='X_only', device='cpu')

print(f"LUT entries: {len(lut)}")
for i, pattern in lut.items():
    print(f"Syndrome bit {i}: {pattern}")
```

### 3. Stim 회로 확인

```python
from qec.simulation.quantum_simulator import StimSurfaceCodeSimulator

sim = StimSurfaceCodeSimulator(distance=3, physical_error_rate=0.09)

# 회로 정보
print(f"Num qubits: {sim.circuit.num_qubits}")
print(f"Num detectors: {sim.circuit.num_detectors}")
print(f"Num observables: {sim.circuit.num_observables}")

# 회로 다이어그램 (간단한 버전만)
# print(sim.circuit)  # 전체 회로 출력 (길 수 있음)
```

---

## 🚀 성능 최적화

### GPU 사용

```bash
# Transformer 디코딩은 GPU에서 빠름
python run_full_workflow.py --device cuda
```

### 배치 처리

```python
# 대량 신드롬 한 번에 디코딩
syndromes = np.random.randint(0, 2, (10000, 8))
predictions = decoder.decode_batch(syndromes)  # 배치 처리
```

### Stim 병렬화

```python
# Stim은 내부적으로 병렬 최적화됨
# 샷 수를 늘려도 선형 증가 안 함
result_10k = sim.sample_syndromes(10000)    # ~1초
result_100k = sim.sample_syndromes(100000)  # ~5초 (10배 아님)
```

---

## 📝 요약

**패키지 구조**:
- `core`: 기본 코드 생성
- `models`: 신경망 디코더
- `decoders`: 전통적 디코더
- `training`: 모델 학습
- **`evaluation`**: 평가 (빠른 + 현실적) ← 새로 추가
- **`simulation`**: Stim 양자 시뮬레이터 ← 새로 추가

**사용 방법**:
1. **간단**: `python run_full_workflow.py`
2. **세밀한 제어**: 모듈별 직접 import
3. **커스텀**: 기존 클래스 상속/확장

이제 연구용 수준의 양자 오류 정정 프레임워크를 갖추셨습니다! 🎉

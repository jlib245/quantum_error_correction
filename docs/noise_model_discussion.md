# Noise Model Discussion

## 현재 사용 중인 노이즈 모델

### Code Capacity Noise Model
- 데이터 qubit에만 Pauli 에러 (X, Y, Z) 발생
- Stabilizer 측정 회로: 완벽 가정
- Syndrome 측정: 완벽 가정

## Y-Biased Noise 실험의 의미

### MWPM의 구조적 한계
MWPM (Minimum Weight Perfect Matching)은 X syndrome과 Z syndrome을 **독립적으로** 디코딩한다.

```
Y = iXZ (X와 Z가 동시에 발생)

MWPM 디코딩:
├── X syndrome → Z error 추정 (독립)
└── Z syndrome → X error 추정 (독립)

→ Y 오류의 X-Z 상관관계를 무시
```

### Neural Network의 장점
- Syndrome 전체를 입력으로 받아 X-Z 상관관계 학습 가능
- Y-biased noise에서 MWPM 대비 성능 우위 예상

## Limitation

### 물리적 정당성
Y-biased noise가 실제로 많이 발생하는 상황:
- **Two-qubit 게이트 오류** (CNOT, CZ 등)
- **Amplitude damping** (T1 decay)
- **Cross-talk** (qubit 간 간섭)

그러나 현재 code capacity 모델에서는 게이트 오류를 고려하지 않으므로, Y-biased noise의 물리적 근거가 약하다.

### 노이즈 모델 비교

| 모델 | 데이터 에러 | 게이트 에러 | 측정 에러 | 현실성 |
|------|------------|------------|----------|--------|
| **Code Capacity** (현재) | O | X | X | 낮음 |
| Phenomenological | O | X | O | 중간 |
| **Circuit-level** | O | O | O | 높음 |

## 실험의 의의

1. **학술적 의미**: MWPM의 상관 오류 취약성 검증
2. **방법론적 의미**: Neural network 디코더의 상관 오류 학습 능력 평가
3. **한계**: Code capacity 수준의 평가이며, 실제 하드웨어 적용을 위해서는 circuit-level 평가 필요

## 향후 연구 방향

### Circuit-level Noise Model
- Stim 라이브러리를 활용한 실제 stabilizer 측정 회로 시뮬레이션
- 게이트 오류, 측정 오류 포함
- 이 환경에서 Y 성분 오류가 자연스럽게 발생

### 기타 노이즈 모델
- **Coherent error**: 작은 회전 오류 (systematic)
- **Leakage**: Computational space 이탈
- **Correlated error**: Qubit 간 상관된 오류

## 참고 문헌 제안

- Y-biased noise와 two-qubit gate 관련: [관련 논문 추가 필요]
- Circuit-level noise model: Stim 논문 (Gidney, 2021)
- MWPM decoder: PyMatching (Higgott, 2021)

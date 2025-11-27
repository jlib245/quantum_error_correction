# Diamond CNN 분석 및 개선

## 0. 구조도

### Before (4채널) - 성능 안 나옴
```
                    Syndrome (24)
                         │
                         ▼
            ┌────────────────────────┐
            │   DiamondGridBuilder   │
            └────────────────────────┘
                         │
         ┌───────┬───────┼───────┬───────┐
         ▼       ▼       ▼       ▼       │
       Ch0     Ch1     Ch2     Ch3       │
      LUT_X   LUT_Z   Z_syn   X_syn      │
        │       │       │       │        │
        └───┬───┘       └───┬───┘        │
            │               │            │
      @ Q positions   @ Stab positions   │
      (25 pixels)      (24 pixels)       │
            │               │            │
            └───────┬───────┘            │
                    ▼                    │
           Grid (4, 9, 9)                │
           77% 빈칸 (-0.5)               │
                    │                    │
                    ▼                    │
         ┌──────────────────┐            │
         │  2x2 Conv → ...  │            │
         │   CNN Layers     │            │
         └──────────────────┘            │
                    │                    │
                    ▼                    │
              Output (4)                 │
                                         │
    문제: Ch2,3 (syndrome)과              │
         Ch0,1 (qubit)이 다른 위치!      │
         CNN이 관계를 직접 배워야 함 ────┘
```

### After (6채널) - 해결
```
                    Syndrome (24)
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ LUT calc │   │ Raw syn  │   │ H.T @ s  │
    │ s@LUT%2  │   │ s_z, s_x │   │aggregate │
    └──────────┘   └──────────┘   └──────────┘
          │              │              │
          ▼              ▼              ▼
       Ch0,1          Ch2,3          Ch4,5
       LUT_X,Z        Z,X syn      Z,X count
          │              │              │
          │              │              │
    @ Q positions  @ Stab pos    @ Q positions
     (25 pixels)   (24 pixels)    (25 pixels)
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                Grid (6, 9, 9)
                         │
                         ▼
              ┌──────────────────┐
              │  2x2 Conv (6→64) │
              │    CNN Layers    │
              └──────────────────┘
                         │
                         ▼
                   Output (4)

    핵심: Ch4,5가 ECC_CNN과 같은 정보 제공!
         → CNN이 이미 모아진 정보도 볼 수 있음
```

### ECC_CNN vs Diamond CNN 입력 비교
```
┌─────────────────────────────────────────────────────────────┐
│                        ECC_CNN                              │
├─────────────────────────────────────────────────────────────┤
│  Input: H.T @ syndrome → (L×L) = 5×5 dense grid            │
│                                                             │
│  ┌─────────────────────┐                                    │
│  │ 2  1  0  1  2 │  ← 각 위치 = "몇 개 syndrome이          │
│  │ 1  3  2  3  1 │     이 qubit을 가리키는지"              │
│  │ 0  2  4  2  0 │                                         │
│  │ 1  3  2  3  1 │  100% 유효, 빈칸 없음                   │
│  │ 2  1  0  1  2 │                                         │
│  └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Diamond CNN (Before)                     │
├─────────────────────────────────────────────────────────────┤
│  Input: 45도 회전 grid → (2L-1)×(2L-1) = 9×9 sparse        │
│                                                             │
│  ┌───────────────────────────────┐                          │
│  │ .  .  .  .  Q  Z  .  .  . │                             │
│  │ .  .  X  Q  X  Q  .  .  . │  Q = qubit (Ch0,1)          │
│  │ .  .  Q  Z  Q  Z  Q  Z  . │  Z = Z-stab (Ch2)           │
│  │ X  Q  X  Q  X  Q  X  Q  . │  X = X-stab (Ch3)           │
│  │ Q  Z  Q  Z  Q  Z  Q  Z  Q │  . = 빈칸 (-0.5)            │
│  │ .  Q  X  Q  X  Q  X  Q  X │                             │
│  │ .  Z  Q  Z  Q  Z  Q  .  . │  77% 빈칸!                  │
│  │ .  .  .  Q  X  Q  X  .  . │                             │
│  │ .  .  .  Z  Q  .  .  .  . │                             │
│  └───────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Diamond CNN (After)                      │
├─────────────────────────────────────────────────────────────┤
│  Ch0-3: 기존과 동일 (sparse)                                │
│  Ch4-5: H.T @ syndrome @ Q positions (aggregated!)         │
│                                                             │
│  Ch4 예시 (Z count at Q positions):                        │
│  ┌───────────────────────────────┐                          │
│  │ .  .  .  .  2  .  .  .  . │                             │
│  │ .  .  .  1  .  3  .  .  . │  ← ECC_CNN과 같은 정보가    │
│  │ .  .  0  .  2  .  4  .  . │     Q 위치에 배치됨         │
│  │ .  1  .  3  .  2  .  3  . │                             │
│  │ 2  .  0  .  2  .  4  .  2 │                             │
│  │ .  1  .  3  .  2  .  3  . │                             │
│  │ .  .  0  .  2  .  4  .  . │                             │
│  │ .  .  .  1  .  3  .  .  . │                             │
│  │ .  .  .  .  2  .  .  .  . │                             │
│  └───────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### 모델 구조 비교
```
┌────────────────────────┬────────────────────────┬────────────────────────┐
│      ECC_CNN           │    Diamond (Before)    │    Diamond (After)     │
├────────────────────────┼────────────────────────┼────────────────────────┤
│                        │                        │                        │
│  Syndrome (24)         │  Syndrome (24)         │  Syndrome (24)         │
│       │                │       │                │       │                │
│       ▼                │       ▼                │       ▼                │
│  H.T @ s               │  Grid Builder          │  Grid Builder          │
│       │                │       │                │       │                │
│       ▼                │       ▼                │       ▼                │
│  (2, 5, 5)             │  (4, 9, 9)             │  (6, 9, 9)             │
│  100% dense            │  23% valid             │  23% + aggregated      │
│       │                │       │                │       │                │
│       ▼                │       ▼                │       ▼                │
│  3×3 Conv ×4           │  2×2 Conv ×2           │  2×2 Conv              │
│       │                │  3×3 Conv ×1           │  Dilated Conv          │
│       │                │       │                │  Self-Attention        │
│       ▼                │       ▼                │       │                │
│  Global Pool           │  Global Pool           │       ▼                │
│       │                │       │                │  Global Pool           │
│       ▼                │       ▼                │       │                │
│  FC → 4                │  FC → 4                │       ▼                │
│                        │                        │  FC → 4                │
│  ✓ 잘 됨               │  ✗ 안 됨              │  ? 테스트 필요         │
└────────────────────────┴────────────────────────┴────────────────────────┘
```

---

## 1. Diamond CNN 개요

### 원래 아이디어
- Rotated Surface Code를 **45도 회전**하여 dense grid로 변환
- **2x2 conv**로 자연스럽게 [Q, Z; X, Q] 마름모(diamond) 패턴 캡처
- LUT prediction을 채널에 포함하여 도메인 지식 활용

### Grid 구조 (L=5 기준)
```
   0 1 2 3 4 5 6 7 8
0: . . . . Q Z . . .
1: . . X Q X Q . . .
2: . . Q Z Q Z Q Z .
3: X Q X Q X Q X Q .
4: Q Z Q Z Q Z Q Z Q  ← 중심
5: . Q X Q X Q X Q X
6: . Z Q Z Q Z Q . .
7: . . . Q X Q X . .
8: . . . Z Q . . . .

Q = Qubit, Z = Z-stabilizer, X = X-stabilizer, . = 빈칸
```

---

## 2. 문제점 분석

### 2.1 성능 비교 (L=5, y_ratio=0.3)

| p_error | LUT only | Diamond CNN | ECC_CNN | Transformer | MWPM |
|---------|----------|-------------|---------|-------------|------|
| 0.07 | 33.6% | 10.0% | ~3-4% | 3.4% | 3.8% |
| 0.10 | 43.8% | 16.0% | ~8-9% | 8.9% | 9.1% |
| 0.13 | 50.1% | 23.3% | ~10-11% | ~10.6% | ~11% |

- Diamond CNN은 LUT보다 3배 좋음 (뭔가 배우고 있음)
- 하지만 ECC_CNN/Transformer보다 2-3배 나쁨

### 2.2 핵심 문제: Sparse Grid

**Diamond CNN Grid:**
- 크기: 9×9 = 81 pixels
- 유효 픽셀: 23% (74/324 across 4 channels)
- **77%가 빈칸 (-0.5)**

**ECC_CNN Grid:**
- 크기: 5×5 = 25 pixels
- 유효 픽셀: **100%**
- 빈칸 없음

### 2.3 근본 원인: Aggregation 부재

**ECC_CNN의 입력:**
```python
# H.T @ syndrome = 각 qubit에 대한 "vote count"
z_count = H_z.T @ s_z  # (n_qubits,)
x_count = H_x.T @ s_x  # (n_qubits,)

# 결과: 각 qubit 위치에 "몇 개의 syndrome이 이 qubit을 가리키는지"
# 이미 유용한 feature가 추출되어 있음!
```

**Diamond CNN의 입력:**
```python
# Raw syndrome을 공간에 배치
Ch0: LUT X prediction @ qubit positions
Ch1: LUT Z prediction @ qubit positions
Ch2: Z syndrome @ Z-stabilizer positions  # 다른 위치!
Ch3: X syndrome @ X-stabilizer positions  # 다른 위치!

# 문제: syndrome과 qubit이 다른 위치에 있음
# CNN이 먼저 이 관계를 배워서 "모아야" 함
```

### 2.4 왜 2x2 conv가 충분하지 않은가

2x2 conv는 4채널을 동시에 보지만:
```
Window (3,0)-(4,1):
  Ch0: [., Q1_lut, Q0_lut, .]
  Ch2: [., ., ., Z0]

→ Z0 syndrome과 Q0, Q1의 LUT는 같은 window에 있음
→ 하지만 Z0가 측정하는 모든 qubit이 같은 window에 없음
→ 여러 window를 종합해야 하는데, 초기 layer에선 불가능
```

---

## 3. 해결 방법

### 3.1 Aggregated Syndrome 채널 추가 (4ch → 6ch)

```python
# DiamondGridBuilder.forward() 수정
def forward(self, syndrome):
    ...
    # 기존 4채널
    Ch0: LUT X prediction @ Q positions
    Ch1: LUT Z prediction @ Q positions
    Ch2: Z syndrome @ Z-stab positions
    Ch3: X syndrome @ X-stab positions

    # 새로 추가된 2채널 (aggregated)
    z_count = s_z @ H_z  # (B, n_qubits)
    x_count = s_x @ H_x  # (B, n_qubits)

    Ch4: H_z.T @ s_z @ Q positions  ← "이 qubit 주변 Z syndrome 몇 개?"
    Ch5: H_x.T @ s_x @ Q positions  ← "이 qubit 주변 X syndrome 몇 개?"
```

### 3.2 변경된 Grid 구조

```
Before (4 channels):
- Ch0,1: LUT prediction (qubit 위치)
- Ch2,3: raw syndrome (stabilizer 위치) ← 다른 위치!

After (6 channels):
- Ch0,1: LUT prediction (qubit 위치)
- Ch2,3: raw syndrome (stabilizer 위치)
- Ch4,5: aggregated syndrome (qubit 위치) ← 같은 위치! ECC_CNN과 동일한 정보
```

### 3.3 모델 수정

모든 Diamond CNN 모델의 첫 번째 conv 입력 채널 변경:
- `ECC_DiamondCNN`: Conv2d(4→6, 64, ...)
- `ECC_DiamondCNN_Deep`: Conv2d(4→6, 64, ...)
- `ECC_DiamondCNN_Attention`: Conv2d(4→6, 64, ...)

---

## 4. 핵심 인사이트

### 왜 ECC_CNN이 잘 되는가
```
Input: H.T @ syndrome
       ↓
각 qubit에 이미 "aggregated" 정보가 있음
       ↓
CNN은 qubit 간 관계만 배우면 됨
```

### 왜 Diamond CNN이 안 됐는가
```
Input: sparse grid (77% 빈칸)
       ↓
syndrome과 qubit이 다른 위치
       ↓
CNN이 먼저 "모으는" 작업을 해야 함
       ↓
capacity 낭비 + 정보 손실
```

### 해결책
```
Input: sparse grid + aggregated channels
       ↓
Ch4,5에 이미 "모아진" 정보 제공
       ↓
CNN이 raw(Ch2,3) + aggregated(Ch4,5) 둘 다 활용
       ↓
ECC_CNN 수준 성능 기대
```

---

## 5. 파일 변경 내역

### `qec/models/diamond_cnn.py`

1. **DiamondGridBuilder.forward()**: 4채널 → 6채널 출력
2. **ECC_DiamondCNN**: conv1 입력 4 → 6
3. **ECC_DiamondCNN_Deep**: stem 입력 4 → 6
4. **ECC_DiamondCNN_Attention**: stem 입력 4 → 6

---

## 6. 추가 개선 가능성

1. **-0.5 값 처리**: learnable embedding으로 대체
2. **Attention 위치**: 입력단에 cross-attention 추가
3. **Multi-scale fusion**: 다양한 receptive field의 feature 결합
4. **GNN 구조**: grid 대신 graph로 처리

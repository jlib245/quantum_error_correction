# 양자 오류 정정 Transformer 학습 과정 완전 가이드

## 목차
1. [초기화 단계](#step-0-초기화)
2. [데이터 생성](#step-1-데이터-생성)
3. [Forward Pass](#step-2-forward-pass)
4. [Loss 계산 & Backpropagation](#step-3-loss-계산--backpropagation)
5. [Epoch 완료](#step-4-epoch-완료)
6. [전체 학습 루프](#step-5-전체-학습-루프)
7. [테스트 단계](#step-6-테스트-단계)
8. [추론 과정](#step-7-추론-inference)

---

## STEP 0: 초기화

### 1. Surface Code 생성
**위치**: `train_transformer.py:518-536`

```python
# L=3인 Surface Code 생성 예시
Hx, Hz, Lx, Lz = Get_surface_Code(L=3)

# 패리티 체크 행렬 (Parity Check Matrix)
H_z = [[1,1,0,0,0,0,0,0,0],    # Z stabilizer (X 오류 검출)
       [0,1,1,0,0,0,0,0,0],
       ...]

H_x = [[1,0,0,1,0,0,0,0,0],    # X stabilizer (Z 오류 검출)
       [0,1,0,0,1,0,0,0,0],
       ...]

# 논리 연산자 행렬
L_z = [[1,1,1,0,0,0,0,0,0]]    # 논리 Z
L_x = [[1,0,0,1,0,0,1,0,0]]    # 논리 X
```

**Surface Code 구조 (L=3)**:
- **물리 큐빗**: 9개 (3×3 격자)
- **Z Stabilizer**: 4개 (X 오류 검출용)
- **X Stabilizer**: 4개 (Z 오류 검출용)
- **논리 큐빗**: 1개

---

### 2. Lookup Table (LUT) 생성
**위치**: `train_transformer.py:381-382`

```python
x_error_basis_dict = create_surface_code_pure_error_lut(L=3, 'X_only', device)
# 결과:
# {
#   0: tensor([0,0,1,1,1,0,0,0,0]),  # 신드롬 비트 0 → X 오류 패턴
#   1: tensor([0,0,0,1,1,1,0,0,0]),  # 신드롬 비트 1 → X 오류 패턴
#   2: tensor([0,0,0,0,1,1,1,0,0]),
#   3: tensor([0,0,0,0,0,1,1,1,0]),
# }

z_error_basis_dict = create_surface_code_pure_error_lut(L=3, 'Z_only', device)
# {
#   0: tensor([2,2,0,2,2,0,0,0,0]),  # 신드롬 비트 0 → Z 오류 패턴
#   1: tensor([0,2,2,0,2,2,0,0,0]),
#   2: tensor([0,0,2,0,0,2,2,2,0]),
#   3: tensor([0,0,0,2,2,0,2,2,2]),
# }
```

**LUT 생성 알고리즘** (`_get_surface_outer_path_x`):
```python
def _get_surface_outer_path_x(face_row, face_col, fixed_row, L, device):
    """신드롬 위치에서 경계까지 수직 경로 생성"""
    pure_error = torch.zeros(L * L)

    # 신드롬 위치에서 고정된 행까지 수직으로 오류 마킹
    vn_col = face_col - 1 if face_col == L else face_col
    start_row, direction = (up_vn_row, -1) if up_vn_row < fixed_row else (down_vn_row, 1)

    r = start_row
    while 0 <= r < L:
        vn_idx = r * L + vn_col
        pure_error[vn_idx] = 1  # X 오류 마킹
        r += direction

    return pure_error
```

**Z 오류는 수평 경로로 생성** (`_get_surface_outer_path_z`).

---

### 3. 모델 초기화
**위치**: `train_transformer.py:388-398`

```python
model = ECC_Transformer(args, dropout=0)
# 구조:
# - Input: syndrome_length = 8 (L=3일 때)
# - d_model: 128
# - N_dec: 6 (Transformer 레이어 수)
# - h: 16 (Attention head 수)
# - Output: 4 classes (I, X, Z, Y)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
```

**모델 파라미터 수**: ~수십만 개 (L과 d_model에 따라 달라짐)

---

## STEP 1: 데이터 생성 (1개 샘플 상세)

### Dataset.__getitem__() 전체 과정
**위치**: `train_transformer.py:215-240`

---

### 1.1 물리 오류 생성

```python
p = random.choice([0.07, 0.08, 0.09, 0.1, 0.11])  # 예: p=0.09
e_full = generate_noise(p, y_ratio=0.3)
```

**노이즈 모델** (`generate_correlated_noise`):
```python
# Y 오류 비율 30%인 경우
p_Y = 0.09 * 0.3 = 0.027
p_X = 0.09 * 0.35 = 0.0315
p_Z = 0.09 * 0.35 = 0.0315

# 각 큐빗에 대해:
rand = random.rand()
if rand < 0.0315:        # X 오류
    e_x[i] = 1
elif rand < 0.0585:      # Y 오류 (X+Z 동시)
    e_x[i] = 1
    e_z[i] = 1
elif rand < 0.09:        # Z 오류
    e_z[i] = 1
```

**예시 결과** (L=3, 9 qubits):
```python
e_z = [0, 1, 0, 0, 1, 0, 0, 0, 0]  # Z 오류 (큐빗 1, 4)
e_x = [0, 0, 0, 1, 0, 0, 0, 1, 0]  # X 오류 (큐빗 3, 7)
e_full = [0,1,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,1,0]
#        ↑─────Z 성분 (9개)─────↑ ↑─────X 성분 (9개)─────↑
```

---

### 1.2 신드롬 측정

```python
e_z_actual = e_full[:9]   # [0,1,0,0,1,0,0,0,0]
e_x_actual = e_full[9:]   # [0,0,0,1,0,0,0,1,0]

s_z_actual = (H_z @ e_x_actual) % 2
s_x_actual = (H_x @ e_z_actual) % 2
syndrome = torch.cat([s_z_actual, s_x_actual])
```

**계산 예시**:
```python
# Z Stabilizer가 X 오류 검출
H_z = [[1,1,0,0,0,0,0,0,0],
       [0,1,1,0,0,0,0,0,0],
       [0,0,0,1,1,0,0,0,0],
       [0,0,0,0,1,1,0,0,0]]

e_x = [0,0,0,1,0,0,0,1,0]

s_z[0] = (1*0 + 1*0 + 0*0 + ... ) % 2 = 0
s_z[1] = (0*0 + 1*0 + 1*0 + ... ) % 2 = 0
s_z[2] = (0*0 + 0*0 + 0*0 + 1*1 + 1*0 + ...) % 2 = 1  ← 큐빗 3 검출
s_z[3] = (0*0 + ... + 1*1 + ...) % 2 = 1  ← 큐빗 7 검출

s_z = [0, 0, 1, 1]
```

```python
# X Stabilizer가 Z 오류 검출
H_x = [[1,0,0,1,0,0,0,0,0],
       [0,1,0,0,1,0,0,0,0],
       [0,0,1,0,0,1,0,0,0],
       [0,0,0,0,0,0,1,1,0]]

e_z = [0,1,0,0,1,0,0,0,0]

s_x[0] = (1*0 + 0*1 + 0*0 + 1*0 + ...) % 2 = 0
s_x[1] = (0*0 + 1*1 + 0*0 + 0*0 + 1*1 + ...) % 2 = 0  ← 큐빗 1, 4 검출 (짝수)
s_x[2] = 0
s_x[3] = 0

s_x = [0, 0, 0, 0]
```

**최종 신드롬**:
```python
syndrome = [0, 0, 1, 1, 0, 0, 0, 0]  # 8차원
#          ↑──s_z (4)──↑ ↑──s_x (4)──↑
```

---

### 1.3 Simple Decoder C (LUT 활용)

**위치**: `train_transformer.py:165-180`

```python
def simple_decoder_C_torch(syndrome_vector, x_error_basis, z_error_basis, H_z, H_x):
    c_x = torch.zeros(9, dtype=torch.uint8)
    c_z = torch.zeros(9, dtype=torch.uint8)

    s_z = syndrome_vector[:4]  # [0, 0, 1, 1]
    s_x = syndrome_vector[4:]  # [0, 0, 0, 0]

    # s_z를 보고 X 오류 추정 (LUT 룩업)
    for i in range(len(s_z)):
        if s_z[i] == 1 and i in x_error_basis:
            c_x.bitwise_xor_(x_error_basis[i])

    # s_x를 보고 Z 오류 추정 (LUT 룩업)
    for i in range(len(s_x)):
        if s_x[i] == 1 and i in z_error_basis:
            c_z.bitwise_xor_(z_error_basis[i])

    return torch.cat([c_z, c_x])
```

**실제 동작**:
```python
# s_z = [0, 0, 1, 1]에서 비트 2, 3이 켜짐
c_x = zeros(9)

# i=2: s_z[2]=1
c_x ^= x_error_basis_dict[2]
# c_x = [0,0,0,0,0,0,0,0,0] ^ [0,0,0,0,1,1,1,0,0]
# c_x = [0,0,0,0,1,1,1,0,0]

# i=3: s_z[3]=1
c_x ^= x_error_basis_dict[3]
# c_x = [0,0,0,0,1,1,1,0,0] ^ [0,0,0,0,0,1,1,1,0]
# c_x = [0,0,0,0,1,0,0,1,0]  ← 추정된 X 오류

# s_x = [0,0,0,0]이므로 Z 오류 추정 없음
c_z = [0,0,0,0,0,0,0,0,0]

pure_error_C = [c_z, c_x] = [0,0,0,0,0,0,0,0,0, 0,0,0,0,1,0,0,1,0]
```

---

### 1.4 논리 오류 계산

```python
# 실제 오류 XOR 추정 오류 = 논리 오류
l_physical = pure_error_C.long() ^ e_full.long()
```

**계산**:
```python
e_full       = [0,1,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,1,0]
pure_error_C = [0,0,0,0,0,0,0,0,0, 0,0,0,0,1,0,0,1,0]
             XOR ────────────────────────────────────
l_physical   = [0,1,0,0,1,0,0,0,0, 0,0,0,1,1,0,0,0,0]
#              ↑─────l_z (9)─────↑ ↑─────l_x (9)─────↑

l_z_physical = [0,1,0,0,1,0,0,0,0]
l_x_physical = [0,0,0,1,1,0,0,0,0]
```

---

### 1.5 논리 큐빗 플립 확인

```python
# 논리 X 플립 여부
l_x_flip = (L_z @ l_x_physical) % 2

# L_z = [1,1,1,0,0,0,0,0,0]
# l_x = [0,0,0,1,1,0,0,0,0]
# l_x_flip = (1*0 + 1*0 + 1*0 + 0*1 + 0*1 + ...) % 2 = 0

# 논리 Z 플립 여부
l_z_flip = (L_x @ l_z_physical) % 2

# L_x = [1,0,0,1,0,0,1,0,0]
# l_z = [0,1,0,0,1,0,0,0,0]
# l_z_flip = (1*0 + 0*1 + 0*0 + 1*0 + 0*1 + ...) % 2 = 0
```

**최종 레이블**:
```python
true_class_index = (l_z_flip * 2 + l_x_flip).long()
                 = (0 * 2 + 0) = 0  ← I 클래스 (오류 없음)
```

**4가지 클래스 매핑**:
| l_z_flip | l_x_flip | 계산값 | 클래스 | 의미 |
|----------|----------|--------|--------|------|
| 0 | 0 | 0 | **I** | 논리 큐빗 정상 |
| 0 | 1 | 1 | **X** | 논리 X 플립 |
| 1 | 0 | 2 | **Z** | 논리 Z 플립 |
| 1 | 1 | 3 | **Y** | 논리 Y 플립 |

---

### 데이터셋 1개 샘플 완성

```python
return (
    syndrome.float(),      # tensor([0,0,1,1,0,0,0,0]) - 8차원
    true_class_index.cpu() # tensor(0) - I 클래스
)
```

---

## STEP 2: Forward Pass (배치 128개)

### 2.1 DataLoader가 배치 생성
**위치**: `train_transformer.py:414-415`

```python
batch_size = 128

# DataLoader가 128개 샘플 묶음
syndrome_batch = torch.tensor([
    [0,0,1,1,0,0,0,0],  # 샘플 1
    [1,1,0,0,1,0,1,0],  # 샘플 2
    [0,1,0,1,0,1,0,1],  # 샘플 3
    ...                 # 125개 더
])  # shape: (128, 8)

labels_batch = torch.tensor([0, 1, 2, 3, 1, 0, ...])  # shape: (128,)
```

---

### 2.2 Transformer Forward Pass
**위치**: `transformer.py:190-216`

#### Step 1: Input Embedding
```python
x_emb = self.input_embedding(syndrome_batch)
# Linear(syndrome_length=8, d_model=128)
# (128, 8) → (128, 128)

# 예시 (샘플 1):
# syndrome[0] = [0,0,1,1,0,0,0,0]
# x_emb[0] = [0.23, -0.45, 0.87, ..., 0.12]  (128차원 벡터)
```

#### Step 2: Sequence Reshaping & Positional Encoding
```python
x_seq = x_emb.unsqueeze(1)  # (128, 1, 128)
# 시퀀스 길이 1로 만듦 (신드롬 전체를 1개 토큰으로 취급)

x_pos = self.pos_encoder(x_seq)
# Positional encoding 추가
# PE(pos=0, i) = sin(0 / 10000^(2i/128))
# x_pos[0,0,:] = x_seq[0,0,:] + PE(0)
```

#### Step 3: Transformer Encoder (N_dec=6 레이어)
```python
encoded_x = self.encoder(x_pos, mask=None)

# 각 레이어 동작:
for layer_idx in range(6):
    # (1) Multi-Head Attention
    Q = Linear_Q(x_pos)  # (128, 1, 128)
    K = Linear_K(x_pos)  # (128, 1, 128)
    V = Linear_V(x_pos)  # (128, 1, 128)

    # Q, K, V를 16개 head로 분할
    Q = Q.view(128, 1, 16, 8)  # (batch, seq, heads, d_k)
    K = K.view(128, 1, 16, 8)
    V = V.view(128, 1, 16, 8)

    # Attention scores
    scores = (Q @ K.transpose(-2, -1)) / sqrt(8)
    # scores: (128, 16, 1, 1)

    attention_weights = softmax(scores, dim=-1)
    attention_output = attention_weights @ V
    # (128, 16, 1, 8) → concat → (128, 1, 128)

    # (2) Residual + LayerNorm
    x_attn = LayerNorm(x_pos + attention_output)

    # (3) Feed-Forward Network
    ff_output = Linear2(GELU(Linear1(x_attn)))
    # Linear1: (128) → (512)
    # GELU 활성화
    # Linear2: (512) → (128)

    # (4) Residual + LayerNorm
    x_pos = LayerNorm(x_attn + ff_output)

encoded_x = x_pos  # (128, 1, 128)
```

#### Step 4: Classification Head
```python
seq_output = encoded_x.squeeze(1)  # (128, 128)
logits = self.output_classifier(seq_output)
# Linear(128, 4)
# (128, 128) → (128, 4)

# 예시 logits (샘플 1):
logits[0] = [2.3, -0.5, 0.1, -1.2]
#            ↑I   ↑X   ↑Z   ↑Y
# → Softmax 후: [0.89, 0.05, 0.04, 0.02]
# → Predicted class: 0 (I)
```

---

## STEP 3: Loss 계산 & Backpropagation

### 3.1 Cross-Entropy Loss 계산
**위치**: `train_transformer.py:288-291`

```python
loss = model.loss(outputs, labels)
# = nn.CrossEntropyLoss()(logits, labels_batch)
```

**계산 과정** (샘플 1 예시):
```python
logits[0] = [2.3, -0.5, 0.1, -1.2]
label[0] = 0  # I 클래스

# Softmax
exp_logits = [exp(2.3), exp(-0.5), exp(0.1), exp(-1.2)]
           = [9.97, 0.61, 1.11, 0.30]
sum_exp = 12.0
softmax = [9.97/12.0, 0.61/12.0, 1.11/12.0, 0.30/12.0]
        = [0.83, 0.05, 0.09, 0.03]

# Cross-Entropy
loss_sample_0 = -log(softmax[0]) = -log(0.83) = 0.186
```

**배치 전체**:
```python
losses = [0.186, 0.523, 0.089, ..., 0.234]  # 128개
batch_loss = mean(losses) = 0.312
```

---

### 3.2 Backpropagation
**위치**: `train_transformer.py:293-295`

```python
model.zero_grad()    # 기존 그래디언트 초기화
loss.backward()      # 역전파로 그래디언트 계산
optimizer.step()     # 가중치 업데이트 (Adam)
```

**Adam 업데이트 과정**:
```python
# 예시: output_classifier의 한 가중치
W_old = 0.523
grad = ∂loss/∂W = 0.012

# Adam의 1차 모멘트 (기울기 이동평균)
m_t = beta1 * m_{t-1} + (1-beta1) * grad
    = 0.9 * 0.008 + 0.1 * 0.012 = 0.0084

# Adam의 2차 모멘트 (기울기 제곱 이동평균)
v_t = beta2 * v_{t-1} + (1-beta2) * grad^2
    = 0.999 * 0.00005 + 0.001 * 0.000144 = 0.0000501

# 편향 보정
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

# 가중치 업데이트
W_new = W_old - lr * m_hat / (sqrt(v_hat) + epsilon)
      = 0.523 - 0.001 * 0.0084 / (sqrt(0.0000501) + 1e-8)
      = 0.523 - 0.00119
      = 0.52181
```

---

### 3.3 정확도 계산
**위치**: `train_transformer.py:297-302`

```python
_, predicted = torch.max(outputs.data, 1)
# predicted: 각 샘플의 최대 logit 인덱스
# predicted = [0, 1, 2, 0, 1, ...]  (128개)

correct = (predicted == labels).sum().item()
# labels   = [0, 1, 2, 3, 1, ...]
# matches  = [T, T, T, F, T, ...]
# correct = 115

accuracy = correct / batch_size = 115 / 128 = 89.8%
LER = 1 - accuracy = 10.2%  (Logical Error Rate)
```

---

## STEP 4: 1 Epoch 완료

**위치**: `train_transformer.py:277-310`

```python
# 에폭당 100,000 샘플 / 128 batch_size = 781 iterations
cum_loss = 0
cum_ler = 0
cum_samples = 0

for batch_idx in range(781):
    # STEP 2-3 반복
    syndrome, labels = next(train_loader)
    outputs = model(syndrome)
    loss = model.loss(outputs, labels)

    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()

    cum_loss += loss.item() * len(syndrome)
    cum_ler += correct
    cum_samples += len(syndrome)

    # 중간 로깅 (391번째, 781번째 배치)
    if (batch_idx+1) % 390 == 0:
        logging.info(
            f'Batch {batch_idx+1}/781: '
            f'Loss={cum_loss/cum_samples:.5e} '
            f'LER={1 - cum_ler/cum_samples:.3e}'
        )

# Epoch 1 최종 결과:
epoch_loss = cum_loss / cum_samples  # 예: 0.312
epoch_ler = 1 - cum_ler / cum_samples  # 예: 0.102 (10.2%)
```

---

## STEP 5: 전체 학습 루프

**위치**: `train_transformer.py:423-451`

```python
best_loss = float('inf')
patience_counter = 0

for epoch in range(1, 201):  # 최대 200 epochs
    # ========== 1. Train ==========
    loss, _, ler = train(model, device, train_loader, optimizer, epoch, LR)

    # ========== 2. Learning Rate Scheduling ==========
    scheduler.step()
    # Cosine Annealing: LR이 코사인 곡선을 따라 감소
    # LR_t = eta_min + (LR_0 - eta_min) * (1 + cos(π*t/T_max)) / 2
    # LR: 0.001 → 0.0008 → 0.0006 → ... → 1e-6

    # ========== 3. 모델 저장 (매 에폭) ==========
    torch.save(model, os.path.join(args.path, 'best_model'))

    # ========== 4. Best Model 체크 ==========
    if loss < best_loss - args.min_delta:  # 개선됨
        best_loss = loss
        patience_counter = 0
        torch.save(model, os.path.join(args.path, 'final_model'))
        logging.info(f'Model Saved - New best loss: {best_loss:.5e}')
    else:  # 개선 없음
        patience_counter += 1
        logging.info(f'No improvement. Patience: {patience_counter}/{args.patience}')

    # ========== 5. Early Stopping ==========
    if args.patience > 0 and patience_counter >= args.patience:
        logging.info(f'Early stopping at epoch {epoch}')
        break

    # ========== 6. 테스트 (10 epoch마다) ==========
    if epoch % 10 == 0:
        test(model, device, test_loaders, ps_test)
```

**학습 진행 예시**:
```
Epoch 1:   Loss=2.345e+00, LER=4.53e-01  (거의 랜덤)
Epoch 10:  Loss=8.92e-01,  LER=1.25e-01  (학습 시작)
  → Test: p=0.07 LER=5.2e-02, p=0.11 LER=2.3e-01
Epoch 50:  Loss=2.34e-01,  LER=3.2e-02   (빠른 개선)
  → Test: p=0.07 LER=8.4e-03, p=0.11 LER=4.5e-02
Epoch 100: Loss=8.9e-02,   LER=1.1e-02   (수렴 중)
Epoch 150: Loss=4.5e-02,   LER=5.0e-03   (거의 최적)
Epoch 180: No improvement. Patience: 20/20
  → Early stopping triggered!
```

**Learning Rate Schedule**:
```
0.001 ─────────╮
                ╲
                 ╲
                  ╲___________
                              1e-6
Epoch:  1      50     100    150    200
```

---

## STEP 6: 테스트 단계

**위치**: `train_transformer.py:313-344`

```python
def test(model, device, test_loader_list, ps_range_test):
    model.eval()  # 평가 모드 (Dropout 비활성화)
    test_ler_list = []

    with torch.no_grad():  # 그래디언트 계산 비활성화
        for p, test_loader in zip(ps_range_test, test_loader_list):
            correct = 0
            total = 0

            # 각 p에 대해 10,000개 샘플 테스트
            while total < 10000:
                syndrome, labels = next(iter(test_loader))

                outputs = model(syndrome)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels).sum().item()
                total += len(labels)

            LER = 1 - (correct / total)
            test_ler_list.append(LER)
            print(f'Test p={p:.3e}, LER={LER:.3e}')

    mean_LER = np.mean(test_ler_list)
    logging.info(f'Mean LER = {mean_LER:.3e}')
```

**출력 예시** (Epoch 100):
```
Test p=7.00e-02, LER=1.23e-03  (0.123%)
Test p=8.00e-02, LER=2.45e-03  (0.245%)
Test p=9.00e-02, LER=4.78e-03  (0.478%)
Test p=1.00e-01, LER=8.92e-03  (0.892%)
Test p=1.10e-01, LER=1.56e-02  (1.56%)
Mean LER = 5.72e-03  (0.572%)
```

---

## STEP 7: 추론 (Inference)

### 실제 사용 시나리오

```python
# 1. 학습된 모델 로드
model = torch.load('best_model', weights_only=False).to(device)
model.eval()

# 2. 양자 컴퓨터에서 측정된 신드롬
# (예: 실제 Surface Code 실험 결과)
measured_syndrome = torch.tensor([[0, 1, 1, 0, 1, 0, 0, 1]]).float()

# 3. 추론
with torch.no_grad():
    logits = model(measured_syndrome)
    # logits = [[-1.2, 0.3, 3.1, -0.5]]

    probabilities = F.softmax(logits, dim=1)
    # probs = [[0.02, 0.08, 0.89, 0.01]]

    predicted_class = torch.argmax(logits, dim=1)
    # predicted = [2]  ← Z 오류!

# 4. 결과 해석
class_names = ['I', 'X', 'Z', 'Y']
predicted_error = class_names[predicted_class.item()]
confidence = probabilities[0, predicted_class].item()

print(f'Predicted Error: {predicted_error}')
print(f'Confidence: {confidence:.2%}')
print(f'Probabilities: I={probabilities[0,0]:.2%}, '
      f'X={probabilities[0,1]:.2%}, '
      f'Z={probabilities[0,2]:.2%}, '
      f'Y={probabilities[0,3]:.2%}')
```

**출력**:
```
Predicted Error: Z
Confidence: 89.23%
Probabilities: I=1.52%, X=7.84%, Z=89.23%, Y=1.41%
```

### 양자 컴퓨터에서의 실제 적용

```python
# 1. 양자 회로 실행
qc.measure_stabilizers()  # Stabilizer 측정
syndrome = get_syndrome_from_measurement()

# 2. 신드롬 → 논리 오류 예측
predicted_logical_error = model.predict(syndrome)

# 3. 오류 보정 적용
if predicted_logical_error == 'X':
    qc.apply(LogicalX())  # 논리 X 게이트로 보정
elif predicted_logical_error == 'Z':
    qc.apply(LogicalZ())
elif predicted_logical_error == 'Y':
    qc.apply(LogicalY())
# else: I (아무것도 안 함)

# 4. 양자 계산 계속 진행
qc.continue_computation()
```

---

## 핵심 인사이트

### 1. 왜 Transformer가 잘 작동하나?

#### (1) 신드롬의 공간적 상관관계
```
Surface Code 격자 구조:
□─■─□     신드롬 비트들이 인접한 큐빗을 공유
│ │ │     → Attention이 이 관계를 학습
■─□─■
│ │ │
□─■─□
```

#### (2) Multi-Head Attention의 역할
- **Head 1**: Z stabilizer 간의 관계 학습
- **Head 2**: X stabilizer 간의 관계 학습
- **Head 3-16**: 복잡한 오류 패턴 (Y 오류, 다중 오류 등)

#### (3) 다양한 오류 확률 동시 학습
```python
ps_train = [0.07, 0.08, 0.09, 0.1, 0.11]
# → 모델이 여러 노이즈 레벨에 일반화
```

---

### 2. Simple Decoder C의 역할

#### (1) 기본 추정 제공
```
신드롬 → LUT → 기본 오류 패턴
```

#### (2) 학습 난이도 감소
```
모델이 학습해야 할 것:
  ❌ 신드롬 → 직접 논리 오류 (어려움)
  ✅ 신드롬 → 추정 오류의 보정 (쉬움)
```

#### (3) Surface Code 기하학 활용
- LUT는 Surface Code의 토폴로지 성질 반영
- 경계까지의 최단 경로 = 물리적으로 타당한 추정

---

### 3. 4-클래스 분류의 의미

| 클래스 | 논리 오류 | 물리적 보정 | 양자 게이트 |
|--------|-----------|-------------|-------------|
| **I (0)** | 없음 | 불필요 | 항등 연산 |
| **X (1)** | 논리 X 플립 | Logical X | $\bar{X}$ |
| **Z (2)** | 논리 Z 플립 | Logical Z | $\bar{Z}$ |
| **Y (3)** | 논리 Y 플립 | Logical Y | $\bar{Y} = \bar{X}\bar{Z}$ |

**중요**:
- 물리 큐빗 오류를 보정하는 것이 **아님**
- 논리 큐빗 상태를 보정하는 것

---

### 4. 학습 데이터 특성

#### 에폭당 고정 샘플 수
```python
# 100,000 샘플/에폭 (배치 크기 무관)
# → 일관된 학습 진행 추적 가능
```

#### 데이터 다양성
```python
# 각 샘플마다:
# - 랜덤 오류 확률 p 선택
# - 랜덤 오류 위치
# - 랜덤 오류 타입 (X/Z/Y)
# → 과적합 방지
```

---

### 5. 성능 지표: LER (Logical Error Rate)

#### 정의
```
LER = (논리 오류 예측 실패 횟수) / (전체 테스트 샘플)
```

#### 물리적 의미
- **낮은 LER**: 논리 큐빗이 안정적으로 보호됨
- **높은 LER**: 오류 정정 실패 → 양자 계산 실패

#### Threshold 개념
```
p < p_threshold: LER 감소 (코드 효과적)
p > p_threshold: LER 증가 (코드 무력화)

Surface Code threshold: ~11% (이론값)
```

---

## 전체 파이프라인 요약

```
┌─────────────────────────────────────────────────────────┐
│ STEP 0: 초기화                                            │
│ - Surface Code 생성 (H_z, H_x, L_z, L_x)                 │
│ - LUT 생성 (신드롬 → 순수 오류)                           │
│ - Transformer 모델 초기화                                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 1: 데이터 생성 (1 샘플)                              │
│ 1. 물리 오류 생성 (노이즈 모델)                            │
│ 2. 신드롬 측정 (H @ error)                                │
│ 3. Simple Decoder C (LUT 활용)                           │
│ 4. 논리 오류 계산 (actual ⊕ estimated)                    │
│ 5. 레이블 생성 (I/X/Z/Y)                                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Forward Pass (배치 128)                          │
│ 1. Input Embedding (8 → 128)                            │
│ 2. Positional Encoding                                  │
│ 3. Transformer Encoder (6 layers)                       │
│    - Multi-Head Attention (16 heads)                    │
│    - Feed-Forward Network                               │
│ 4. Classification Head (128 → 4)                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Loss & Backprop                                 │
│ 1. Cross-Entropy Loss 계산                               │
│ 2. Backpropagation (그래디언트 계산)                       │
│ 3. Adam 업데이트                                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 4-5: Epoch 반복 (200회)                             │
│ - Learning Rate Scheduling                              │
│ - Best Model 저장                                         │
│ - Early Stopping                                        │
│ - 주기적 테스트                                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 6-7: 평가 & 추론                                     │
│ - 다양한 p에 대해 LER 측정                                 │
│ - 새로운 신드롬에 대한 예측                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 참고: 주요 하이퍼파라미터

```python
# 코드 파라미터
L = 3                    # Surface Code 크기
n_qubits = L * L = 9     # 물리 큐빗 수
syndrome_len = 8         # 신드롬 길이

# 모델 파라미터
d_model = 128            # Transformer 차원
N_dec = 6                # Encoder 레이어 수
h = 16                   # Attention head 수
d_ff = 512               # FFN 차원 (d_model * 4)

# 학습 파라미터
lr = 0.001               # 초기 learning rate
batch_size = 128         # 배치 크기
epochs = 200             # 최대 에폭 수
samples_per_epoch = 100000  # 에폭당 샘플 수

# 노이즈 파라미터
ps_train = [0.07, 0.08, 0.09, 0.1, 0.11]
y_ratio = 0.3            # Y 오류 비율

# Early Stopping
patience = 20            # 개선 없을 때 대기 에폭
min_delta = 0.0          # 개선 최소 임계값
```

---

## 마치며

이 문서는 양자 오류 정정 Transformer 모델의 **완전한 학습 파이프라인**을 다룹니다.

핵심 요약:
1. Surface Code가 오류를 신드롬으로 변환
2. LUT 기반 Simple Decoder가 기본 추정 제공
3. Transformer가 신드롬 → 논리 오류 매핑 학습
4. 4-클래스 분류로 I/X/Z/Y 예측
5. LER 지표로 성능 평가

이 접근법은 **전통적인 디코더(MWPM 등)보다 빠르고 유연**하며, 다양한 노이즈 모델에 적응 가능합니다.

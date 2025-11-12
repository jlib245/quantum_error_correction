# QEC 학습 하드웨어별 최적화 가이드

Google Colab 및 로컬 환경에서 다양한 하드웨어를 사용한 양자 오류 정정 모델 학습 최적 설정

---

## 📋 목차

- [하드웨어 개요](#하드웨어-개요)
- [Google Colab GPUs](#google-colab-gpus)
  - [Tesla T4 (Free)](#tesla-t4-16gb---colab-free)
  - [L4 GPU](#l4-24gb---colab)
  - [A100 GPU (Pro+)](#a100-40gb---colab-pro)
- [로컬 GPU](#로컬-gpu)
  - [NVIDIA RTX 3070](#nvidia-rtx-3070-8gb)
  - [Intel Arc A350M](#intel-arc-a350m-4gb)
- [CPU](#cpu-설정)
- [성능 비교표](#성능-비교표)
- [빠른 참조](#빠른-참조)

---

## 하드웨어 개요

| 하드웨어 | 메모리 | 환경 | 접근성 | 추천 용도 |
|---------|--------|------|--------|----------|
| **Tesla T4** | 16GB | Colab Free | ⭐⭐⭐⭐⭐ | 빠른 테스트 |
| **L4** | 24GB | Colab | ⭐⭐⭐⭐ | 중간 규모 실험 |
| **A100** | 40GB | Colab Pro+ | ⭐⭐⭐ | 본격 연구 |
| **RTX 3070** | 8GB | 로컬 | ⭐⭐⭐⭐ | 개인 개발 |
| **Arc A350M** | 4GB | 로컬 노트북 | ⭐⭐⭐⭐⭐ | 이동 중 개발 |
| **CPU** | 시스템 RAM | 모든 환경 | ⭐⭐⭐⭐⭐ | 최소 테스트 |

> 💡 **Colab TPU (v5e, v6e)**: PyTorch 지원이 제한적이므로 이 가이드에서는 제외합니다. JAX/TensorFlow 사용 시 고려하세요.

---

## Google Colab GPUs

> **Colab GPU 선택**: 런타임 → 런타임 유형 변경 → 하드웨어 가속기 선택

### Tesla T4 (16GB) - Colab Free

**Turing 아키텍처, 무료로 사용 가능**

#### L=3 (9 qubits, 8 syndromes) ✅ 빠른 테스트

**Transformer:**
```bash
python -m qec.training.train_transformer -L 3 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 512 \
  --test_batch_size 1024 \
  --workers 2 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 3 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 1024 \
  --test_batch_size 2048 \
  --workers 2 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 15-20분, FFNN 8-12분
- 💾 **VRAM**: ~3-4GB
- 🚀 **속도**: CPU 대비 3-4배

#### L=5 (25 qubits, 24 syndromes) ✅ 적합

**Transformer:**
```bash
python -m qec.training.train_transformer -L 5 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 256 \
  --test_batch_size 512 \
  --workers 2 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 5 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 512 \
  --test_batch_size 1024 \
  --workers 2 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 1-1.5시간, FFNN 30-45분
- 💾 **VRAM**: ~5-7GB
- 🚀 **속도**: CPU 대비 4-6배

#### L=7 (49 qubits, 48 syndromes) ⚠️ 메모리 주의

**Transformer:**
```bash
python -m qec.training.train_transformer -L 7 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 128 \
  --test_batch_size 256 \
  --workers 2 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 7 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 256 \
  --test_batch_size 512 \
  --workers 2 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 3-4시간, FFNN 1.5-2시간
- 💾 **VRAM**: ~8-10GB
- ⚠️ **경고**: 메모리 부족 가능, batch_size 64로 줄이기

---

### L4 (24GB) - Colab

**Ada Lovelace 아키텍처, T4의 후속 모델**

#### L=3 (9 qubits, 8 syndromes) ✅ 매우 빠름

**Transformer:**
```bash
python -m qec.training.train_transformer -L 3 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 1024 \
  --test_batch_size 2048 \
  --workers 4 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 3 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 2048 \
  --test_batch_size 4096 \
  --workers 4 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 10-12분, FFNN 5-7분
- 💾 **VRAM**: ~3-4GB
- 🚀 **속도**: T4 대비 1.3-1.5배 빠름

#### L=5 (25 qubits, 24 syndromes) ✅✅ 최적

**Transformer:**
```bash
python -m qec.training.train_transformer -L 5 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 512 \
  --test_batch_size 1024 \
  --workers 4 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 5 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 1024 \
  --test_batch_size 2048 \
  --workers 4 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 35-50분, FFNN 20-25분
- 💾 **VRAM**: ~6-8GB
- 🚀 **속도**: T4 대비 1.5-2배 빠름
- 🎯 **추천**: 연구용 최적 설정

#### L=7 (49 qubits, 48 syndromes) ✅ 여유있음

**Transformer:**
```bash
python -m qec.training.train_transformer -L 7 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 256 \
  --test_batch_size 512 \
  --workers 4 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 7 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 512 \
  --test_batch_size 1024 \
  --workers 4 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 2-2.5시간, FFNN 1-1.5시간
- 💾 **VRAM**: ~10-12GB
- 🚀 **속도**: T4 대비 1.5-2배 빠름
- 💾 **장점**: 24GB 메모리로 여유롭게 실험 가능

---

### A100 (40GB) - Colab Pro+

**Ampere 아키텍처, 최고 성능의 데이터센터 GPU**

#### L=3 (9 qubits, 8 syndromes) 🚀 초고속

**Transformer:**
```bash
python -m qec.training.train_transformer -L 3 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 2048 \
  --test_batch_size 4096 \
  --workers 4 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 3 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 4096 \
  --test_batch_size 8192 \
  --workers 4 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 5-8분, FFNN 3-5분
- 💾 **VRAM**: ~4-5GB
- 🚀 **속도**: T4 대비 2-3배 빠름

#### L=5 (25 qubits, 24 syndromes) ✅✅✅ 최고

**Transformer:**
```bash
python -m qec.training.train_transformer -L 5 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 1024 \
  --test_batch_size 2048 \
  --workers 4 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 5 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 2048 \
  --test_batch_size 4096 \
  --workers 4 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 20-30분, FFNN 12-18분
- 💾 **VRAM**: ~8-10GB
- 🚀 **속도**: T4 대비 3-4배 빠름
- 🎯 **최적**: 본격 연구/논문용

#### L=7 (49 qubits, 48 syndromes) ✅ 매우 빠름

**Transformer:**
```bash
python -m qec.training.train_transformer -L 7 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 512 \
  --test_batch_size 1024 \
  --workers 4 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 7 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 1024 \
  --test_batch_size 2048 \
  --workers 4 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 1-1.5시간, FFNN 40-60분
- 💾 **VRAM**: ~12-16GB
- 🚀 **속도**: T4 대비 3-4배 빠름
- 💰 **비용**: Colab Pro+ 필요

---

## 로컬 GPU

### NVIDIA RTX 3070 (8GB)

**Ampere 아키텍처, 소비자용 GPU의 스위트스팟**

#### L=3 (9 qubits, 8 syndromes) ✅ 빠름

**Transformer:**
```bash
python -m qec.training.train_transformer -L 3 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 1024 \
  --test_batch_size 2048 \
  --workers 4 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 3 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 2048 \
  --test_batch_size 4096 \
  --workers 4 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 10-15분, FFNN 5-8분
- 💾 **VRAM**: ~3-4GB
- 🚀 **속도**: T4와 비슷

#### L=5 (25 qubits, 24 syndromes) ✅✅ 권장

**Transformer:**
```bash
python -m qec.training.train_transformer -L 5 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 512 \
  --test_batch_size 1024 \
  --workers 4 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 5 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 1024 \
  --test_batch_size 2048 \
  --workers 4 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 40-60분, FFNN 20-30분
- 💾 **VRAM**: ~5-6GB
- 🚀 **속도**: CPU 대비 5-8배
- 🎯 **추천**: 개인 연구용 최적

#### L=7 (49 qubits, 48 syndromes) ⚠️ 메모리 한계

**Transformer:**
```bash
python -m qec.training.train_transformer -L 7 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 256 \
  --test_batch_size 512 \
  --workers 4 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 7 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 384 \
  --test_batch_size 768 \
  --workers 4 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 2-3시간, FFNN 1-1.5시간
- 💾 **VRAM**: ~7-8GB (거의 한계)
- ⚠️ **경고**: 메모리 부족 시 batch_size 128로 줄이기
- 📝 **노트**: 8GB VRAM이 빠듯함

---

### Intel Arc A350M (4GB)

**Xe HPG 아키텍처, 노트북용 GPU**

> 💡 **Intel XPU 사용**: Intel Extension for PyTorch 필요
> ```bash
> pip install intel-extension-for-pytorch
> ```

#### L=3 (9 qubits, 8 syndromes) ⚠️ CPU 사용 권장

**Transformer:**
```bash
python -m qec.training.train_transformer -L 3 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 256 \
  --test_batch_size 512 \
  --workers 2 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 3 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 512 \
  --test_batch_size 1024 \
  --workers 2 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 25-35분, FFNN 12-18분
- 💾 **VRAM**: ~1-2GB
- 🐌 **속도**: CPU 대비 1-1.5배 (오버헤드로 큰 차이 없음)
- ⚠️ **경고**: L=3는 너무 작아서 XPU 이점 없음

#### L=5 (25 qubits, 24 syndromes) ✅ 권장

**Transformer:**
```bash
python -m qec.training.train_transformer -L 5 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 256 \
  --test_batch_size 512 \
  --workers 2 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 5 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 384 \
  --test_batch_size 768 \
  --workers 2 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 1.5-2.5시간, FFNN 50-80분
- 💾 **VRAM**: ~2-3GB
- 🚀 **속도**: CPU 대비 2-3배
- 🎯 **추천**: XPU 효과가 나타나는 최소 크기

#### L=7 (49 qubits, 48 syndromes) ⚠️ 메모리 한계

**Transformer:**
```bash
python -m qec.training.train_transformer -L 7 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 128 \
  --test_batch_size 256 \
  --workers 2 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 7 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 192 \
  --test_batch_size 384 \
  --workers 2 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 5-8시간, FFNN 2.5-4시간
- 💾 **VRAM**: ~3-4GB (한계 근접)
- 🚀 **속도**: CPU 대비 3-4배
- ⚠️ **경고**: 4GB VRAM으로 빠듯함, batch_size 64로 줄이기 권장
- 📝 **노트**: 시간이 오래 걸려 비권장, L=5 사용 추천

---

## CPU 설정

**모든 환경에서 사용 가능**

### L=3 (9 qubits, 8 syndromes) ✅ 유일한 선택

**Transformer:**
```bash
python -m qec.training.train_transformer -L 3 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 64 \
  --test_batch_size 128 \
  --workers 2 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 3 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 128 \
  --test_batch_size 256 \
  --workers 2 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 30-45분, FFNN 15-25분
- 💾 **메모리**: ~1-2GB RAM
- 📝 **노트**: CPU는 L=3만 적합

### L=5 (25 qubits, 24 syndromes) ⚠️ 매우 느림

**Transformer:**
```bash
python -m qec.training.train_transformer -L 5 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 32 \
  --test_batch_size 64 \
  --workers 2 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 5 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 64 \
  --test_batch_size 128 \
  --workers 2 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 4-6시간, FFNN 2-3시간
- 💾 **메모리**: ~2-3GB RAM
- ⚠️ **경고**: 매우 느림, GPU 사용 강력 권장

### L=7 (49 qubits, 48 syndromes) ❌ 비권장

**Transformer:**
```bash
python -m qec.training.train_transformer -L 7 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 16 \
  --test_batch_size 32 \
  --workers 2 \
  -y 0.3
```

**FFNN:**
```bash
python -m qec.training.train_ffnn -L 7 \
  --lr 0.001 \
  --epochs 200 \
  --batch_size 32 \
  --test_batch_size 64 \
  --workers 2 \
  -y 0.3
```

- ⏱️ **예상 시간**: Transformer 12-20시간, FFNN 6-10시간
- 💾 **메모리**: ~3-4GB RAM
- ❌ **경고**: 비현실적으로 느림, 절대 비권장

---

## 성능 비교표

### Transformer 학습 시간 (분)

| 하드웨어 | L=3 | L=5 | L=7 | VRAM/RAM |
|---------|-----|-----|-----|----------|
| **A100** | 5-8 🚀 | 20-30 ✅✅✅ | 60-90 ✅ | 40GB |
| **L4** | 10-12 🚀 | 35-50 ✅✅ | 120-150 ✅ | 24GB |
| **T4** | 15-20 ✅ | 60-90 ✅ | 180-240 ⚠️ | 16GB |
| **RTX 3070** | 10-15 ✅ | 40-60 ✅ | 120-180 ⚠️ | 8GB |
| **Arc A350M** | 25-35 ⚠️ | 90-150 ✅ | 300-480 ⚠️ | 4GB |
| **CPU** | 30-45 ✅ | 240-360 ❌ | 720-1200 ❌ | RAM |

### FFNN 학습 시간 (분)

| 하드웨어 | L=3 | L=5 | L=7 | VRAM/RAM |
|---------|-----|-----|-----|----------|
| **A100** | 3-5 🚀 | 12-18 ✅✅✅ | 40-60 ✅ | 40GB |
| **L4** | 5-7 🚀 | 20-25 ✅✅ | 60-90 ✅ | 24GB |
| **T4** | 8-12 ✅ | 30-45 ✅ | 90-120 ⚠️ | 16GB |
| **RTX 3070** | 5-8 ✅ | 20-30 ✅ | 60-90 ⚠️ | 8GB |
| **Arc A350M** | 12-18 ⚠️ | 50-80 ✅ | 150-240 ⚠️ | 4GB |
| **CPU** | 15-25 ✅ | 120-180 ❌ | 360-600 ❌ | RAM |

### 배치 사이즈 권장값

| 하드웨어 | L=3 (Train/Test) | L=5 (Train/Test) | L=7 (Train/Test) |
|---------|-----------------|-----------------|-----------------|
| **A100** | 2048/4096 | 1024/2048 | 512/1024 |
| **L4** | 1024/2048 | 512/1024 | 256/512 |
| **T4** | 512/1024 | 256/512 | 128/256 |
| **RTX 3070** | 1024/2048 | 512/1024 | 256/512 |
| **Arc A350M** | 256/512 | 256/512 | 128/256 |
| **CPU** | 64/128 | 32/64 | 16/32 |

---

## 빠른 참조

### 🎯 상황별 최고 추천

**1. 무료로 빠른 테스트**
- Colab T4 (Free) + L=3 + FFNN → **8-12분** ⭐

**2. 연구용 실험 (비용 효율)**
- Colab L4 + L=5 + Transformer → **35-50분** ⭐⭐⭐

**3. 본격 연구/논문 (최고 성능)**
- Colab A100 + L=5 + Transformer → **20-30분** ⭐⭐⭐⭐⭐

**4. 개인 GPU (RTX 3070)**
- L=5 + Transformer → **40-60분** ⭐⭐⭐⭐

**5. 노트북 (Arc A350M)**
- L=5 + FFNN → **50-80분** (가장 현실적) ⭐⭐⭐

**6. GPU 없이 CPU만**
- L=3 + FFNN → **15-25분** (유일한 선택) ⭐⭐

### 💡 하드웨어 선택 가이드

```
GPU 없음 → CPU + L=3만

무료로 실험 → Colab T4 (Free) + L=3 or L=5

중간 규모 실험 → Colab L4 + L=5 or L=7

본격 연구 → Colab A100 (Pro+) + L=5 or L=7

개인 개발 (로컬) → RTX 3070 + L=5

노트북 개발 → Arc A350M + L=5 (CPU로 L=3도 가능)
```

### 🔥 팁 & 트릭

**1. 특정 디바이스 강제 선택:**
```bash
# CPU 강제 사용 (GPU가 있어도 CPU로 실행)
python -m qec.training.train_transformer -L 3 --device cpu

# CUDA 강제 사용
python -m qec.training.train_transformer -L 3 --device cuda

# XPU 강제 사용 (Intel Arc)
python -m qec.training.train_transformer -L 3 --device xpu

# 자동 선택 (기본값, CUDA > XPU > CPU 순)
python -m qec.training.train_transformer -L 3 --device auto
# 또는 생략
python -m qec.training.train_transformer -L 3
```

**2. GPU 메모리 부족 시:**
```bash
RuntimeError: CUDA out of memory

# 해결: batch_size를 절반으로
--batch_size 256  # 512에서 줄임
--test_batch_size 512  # 1024에서 줄임
```

**3. Colab 세션 유지 (90분 제한 우회):**
```python
# Colab 노트북에서 실행
from IPython.display import Javascript
import time

def keep_alive():
    display(Javascript('''
        function ClickConnect(){
            console.log("Keep alive working...");
            document.querySelector("colab-connect-button").click()
        }
        setInterval(ClickConnect, 60000)
    '''))

keep_alive()
```

**4. Y-ratio 실험:**
```bash
# Y 오류 비율 변경 (0.0 ~ 1.0)
-y 0.0   # Independent noise (X, Z, Y 각 1/3)
-y 0.1   # 10% Y errors
-y 0.2   # 20% Y errors
-y 0.3   # 30% Y errors (기본 권장)
-y 0.5   # 50% Y errors
```

**5. 여러 실험 자동 실행:**
```bash
# Y-ratio 비교 실험
for y in 0.0 0.1 0.2 0.3; do
  python -m qec.training.train_transformer -L 5 \
    --lr 0.001 --epochs 200 \
    --batch_size 512 --workers 4 \
    -y $y
done

# L 비교 실험
for L in 3 5 7; do
  python -m qec.training.train_ffnn -L $L \
    --lr 0.001 --epochs 200 \
    --batch_size 512 --workers 4 \
    -y 0.3
done
```

**6. 학습 중단 후 재개:**
```bash
# 체크포인트 저장 경로 확인
ls Final_Results_QECCT/surface/*/

# 최고 성능 모델 불러오기
--resume_from Final_Results_QECCT/.../best_model
# (현재 구현되지 않음, 추후 추가 예정)
```

---

## 📚 추가 명령어

### 모델 비교 (MWPM vs Neural Networks)

```bash
# Y-ratio 0.3로 MWPM, FFNN, Transformer 비교
python -m qec.training.compare_decoders -L 3 \
  -y 0.3 \
  -p 0.07 0.08 0.09 0.1 0.11 \
  -n 10000 \
  --ffnn_model Final_Results_QECCT/surface/FFNN_Code_L_3/.../final_model \
  --transformer_model Final_Results_QECCT/surface/Transformer_Code_L_3/.../final_model

# MWPM만 테스트
python -m qec.training.compare_decoders -L 5 \
  -y 0.3 \
  -p 0.09 \
  -n 10000 \
  --skip_mwpm False
```

### MWPM 단독 테스트

```bash
python -m qec.training.test_mwpm -L 3 -y 0.3 -p 0.09 -n 10000
python -m qec.training.test_mwpm -L 5 -y 0.2 -p 0.07 0.08 0.09 0.1 0.11 -n 5000
```

### 학습 결과 확인

```bash
# 로그 파일 확인
cat Final_Results_QECCT/surface/Transformer_Code_L_3/.../logging.txt

# 최근 실험 결과 찾기
find Final_Results_QECCT -name "logging.txt" -mtime -1 | xargs ls -lht
```

---

## 🔧 환경별 설치 가이드

### Google Colab (T4, L4, A100)

```bash
# Colab에서 기본 패키지 설치
!pip install torch numpy pymatching tqdm

# 프로젝트 클론 (필요시)
!git clone https://github.com/your-repo/quantum_error_correction.git
%cd quantum_error_correction

# GPU 확인
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 로컬 NVIDIA GPU (RTX 3070 등)

```bash
# CUDA 설치 확인
nvidia-smi

# PyTorch 설치 (CUDA 11.8 예시)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 프로젝트 의존성 설치
pip install numpy pymatching tqdm

# 또는 pyproject.toml 사용
pip install -e .
```

### Intel Arc GPU (A350M 등)

```bash
# Intel Extension for PyTorch 설치
pip install torch numpy pymatching tqdm
pip install intel-extension-for-pytorch

# 또는 pyproject.toml의 intel 옵션
pip install -e ".[intel]"

# XPU 확인
python -c "import torch; import intel_extension_for_pytorch as ipex; print('XPU available:', torch.xpu.is_available())"
```

---

## ❓ FAQ

**Q1: Colab에서 어떤 GPU를 받을지 모르겠어요**
- Free: 보통 T4, 가끔 K80
- Pro: T4 또는 P100
- Pro+: A100 또는 V100

**Q2: L=3, L=5, L=7 중 무엇을 선택해야 하나요?**
- L=3: 빠른 테스트/검증용
- L=5: 연구/논문용 최적 (가장 권장)
- L=7: 대규모 실험/벤치마크

**Q3: batch_size를 얼마로 설정해야 하나요?**
- GPU 메모리가 허락하는 한 크게
- 위 배치 사이즈 권장값 표 참조
- 메모리 부족 시 절반으로 줄이기

**Q4: epochs 200이 너무 많지 않나요?**
- 200은 충분한 수렴을 위한 값
- 빠른 테스트는 50-100으로 줄여도 됨
- Early stopping은 현재 미구현

**Q5: lr 0.001이 최적인가요?**
- Transformer + Adam에는 0.001 권장
- 0.01은 너무 큼 (발산 위험)
- 실험적으로 0.0005~0.002 테스트 가능

**Q6: Arc A350M에서 왜 느린가요?**
- L=3는 너무 작아서 XPU 오버헤드 > 이득
- L=5부터 XPU 효과 나타남
- Intel Extension PyTorch 최신 버전 사용 권장

**Q7: TPU는 사용할 수 없나요?**
- 현재 코드는 PyTorch 기반
- TPU는 JAX/TensorFlow에 최적화
- PyTorch/XLA로 TPU 사용 가능하나 성능 보장 못함

---

**최종 업데이트**: 2025-01-12
**버전**: 2.0
**작성자**: QEC Project Team


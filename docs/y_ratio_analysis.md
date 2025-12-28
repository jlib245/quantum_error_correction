# Y-Ratio Robustness Analysis

## 실험 설정
- **Code Distance**: L = 3, 5, 7
- **Y-ratio**: 33% (학습), 50%, 75%, 100% (테스트)
- **Physical Error Rate**: 0.07 ~ 0.13
- **Test Shots**: 10,000 per error rate

---

## 1. MWPM의 Y-ratio 취약성

Y-ratio 증가 시 MWPM 성능 급락:

```
L=3: 13.6% → 17.7% → 21.7% (60% 악화)
L=5: 12.3% → 17.4% → 22.4% (82% 악화)
L=7: 17.2% → 23.6%         (37% 악화)
```

**원인**: X/Z 독립 디코딩 → Y(=XZ) 상관관계 무시

---

## 2. NN 모델의 Robustness

NN은 Y-ratio 변화에도 안정적:

```
Transformer L=5: 7.1% → 6.5% → 6.7% (거의 일정)
ViT L=3:         9.4% → 9.3% → 8.8% (오히려 개선!)
```

**원인**: Syndrome 전체를 보고 X-Z 상관관계 학습

---

## 3. 개선 효과 (Y=100%, p=0.10)

| L | Best NN | MWPM | 개선율 |
|---|---------|------|--------|
| 3 | ViT 8.8% | 21.7% | **-59%** |
| 5 | Transformer 6.7% | 22.4% | **-70%** |
| 7 | ViT_LUT 7.3% | 23.6% | **-69%** |

**→ NN이 MWPM 대비 2~3배 좋음**

---

## 4. 모델 순위 (Y=100%)

**L=3**: ViT > Transformer > FFNN > LUT > ViT_LUT > CNN

**L=5**: Transformer > ViT > ViT_LUT > CNN > LUT > FFNN

**L=7**: ViT_LUT > Transformer > CNN ≈ ViT > LUT > FFNN

---

## 5. Generalization

- 학습: Y=33%
- 테스트: Y=100% (완전 다른 분포)
- **결과: 성능 유지!**

→ **NN이 특정 패턴 암기가 아닌 일반적 디코딩 전략 학습**

---

## 6. 핵심 발견

1. **Y-ratio↑ → MWPM 급락, NN 안정**
2. **Code distance↑ → NN 우위 확대**
3. **Best**: ViT_LUT_Concat (L=7), Transformer (L=5)
4. **최대 70% 개선** (MWPM 대비)

---

## 7. Limitations

1. **Code Capacity 모델**: Circuit-level noise에서 추가 검증 필요
2. **Y-bias 물리적 근거**: Code capacity에서는 약함, circuit-level에서 2-qubit 게이트 오류로 자연 발생
3. **Surface code만 평가**: 다른 QEC 코드 일반화 필요

---

## 8. 결론

Y-biased noise 환경에서 Neural Network 디코더가 MWPM 대비 **최대 70% 성능 향상**을 보이며, 학습하지 않은 노이즈 분포에서도 **robustness**를 유지함을 확인.

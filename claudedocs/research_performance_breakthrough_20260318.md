# 0.05 벽 돌파 전략 리서치 리포트

**날짜**: 2026-03-18
**현재 상태**: CV 0.0374 / LB 0.0586 (99위→개선 중)
**목표**: LB 0.01 이하 (1위 0.00782)

---

## Executive Summary

**0.05 벽의 정체는 도메인 시프트**. train(고정 조명/카메라)과 test(랜덤 조명/카메라)의 분포 차이가 병목.
Video Teacher KD로 학습 품질은 충분히 개선됨(CV 0.037). 이제 **일반화 갭**을 줄여야 한다.

3가지 전략을 **동시에** 적용하는 것이 가장 효과적:
1. **Dev 기준 Calibration** (즉시 적용, 코드 변경 최소)
2. **Multi-Backbone Ensemble** (가장 큰 실질 효과)
3. **Domain Adversarial Training (DANN)** (근본 해결)

---

## 1. 문제 분석: CV vs LB 갭

```
CV LogLoss:  0.0374  (train+dev 혼합 검증)
LB LogLoss:  0.0586  (test = dev 분포)
갭: 1.57배
```

### 왜 갭이 발생하는가?
- **Train 1000개**: 고정 조명, 고정 카메라 위치 → "실험실 환경"
- **Dev 100개 / Test 1000개**: 랜덤 조명, 랜덤 카메라 → "실제 환경"
- 모델이 train의 고정된 조명 패턴에 과적합 → test에서 확률 추정 오차 발생

### CV가 낙관적인 이유
- CV fold validation에 **train 분포 샘플이 대다수**(~880/220)
- dev 100개가 5 fold에 나뉘어 ~20개씩만 → dev에 대한 검증력 부족
- CV 성능은 주로 train→train 일반화를 측정

---

## 2. 전략별 분석

### 전략 A: Dev-Only Temperature Scaling (즉시 효과, 난이도 하)

**근거**: test와 dev는 같은 분포. temperature를 dev 기준으로 맞추면 test calibration이 개선됨.

**현재 문제**: Temperature scaling이 fold validation 전체(train+dev)에서 fit
→ train 분포에 맞춰진 temperature → test에서 miscalibration

**방법**:
```python
# 현재: fold validation 전체에서 temperature fit
temp = ts.fit(val_logits, val_labels)  # train+dev 혼합

# 개선: dev 샘플에서만 temperature fit
dev_mask = val_data['source_domain'] == 1
if dev_mask.any():
    temp = ts.fit(val_logits[dev_mask], val_labels[dev_mask])
else:
    temp = ts.fit(val_logits, val_labels)  # fallback
```

**예상 효과**: LB 0.058 → 0.04~0.05 (10~20% 개선)
**확신도**: 중 (dev가 fold당 ~20개라 temperature overfitting 위험 있음)

### 전략 B: Multi-Backbone Ensemble (가장 큰 효과, 난이도 중)

**근거**:
- Kaggle 상위권에서 가장 보편적이고 효과적인 전략 [1]
- 서로 다른 backbone은 도메인 시프트에 다르게 반응 → 앙상블이 robustness 향상
- "uncorrelated predictions"가 핵심: 다양한 아키텍처의 예측은 서로 다른 실수를 함

**방법**:
```
Backbone 1: EfficientNetV2-S (현재, timm IN-21k)
Backbone 2: ConvNeXt-V2-Base (v1에서 LB 0.11)
Backbone 3: SwinV2-Small
Backbone 4: EVA02-Small-patch14-224 (CLIP pretrained)

각 backbone × 5 fold = 20 모델
OOF 기반 scipy.optimize로 가중치 최적화
```

**예상 효과**: LB 0.05 → 0.02~0.03 (40~60% 개선)
**확신도**: 높음 (경험적으로 가장 안정적)

### 전략 C: Domain Adversarial Training - DANN (근본 해결, 난이도 중~상)

**근거**:
- Ganin et al. (JMLR 2016): Gradient Reversal Layer로 domain-invariant features 학습 [2]
- train/dev의 source_domain 레이블이 이미 있음 → 바로 적용 가능
- physics_solution에도 domain_head 코드가 있었으나 기본 OFF

**방법**:
- 현재 DualStreamModelV3의 fused features에 domain classifier 추가
- GRL(Gradient Reversal Layer)로 backbone이 domain을 구분 못하도록 학습
- source_domain=0(train) vs source_domain=1(dev)를 구분하는 adversarial task

```python
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None

# domain_head: Linear(fused_dim, 2)
# grl_lambda: 0 → 1 (epoch 진행에 따라 증가)
domain_loss = CE(domain_head(GRL(feat, grl_lambda)), source_domain)
total_loss += domain_weight * domain_loss
```

**예상 효과**: LB 0.05 → 0.03~0.04 (단독), ensemble과 결합 시 추가 개선
**확신도**: 중~높 (검증된 기법이지만 하이퍼파라미터 튜닝 필요)

### 전략 D: Test-Time Training / Adaptation (추가 개선, 난이도 상)

**근거**:
- TTT는 test 샘플 자체를 이용해 모델을 미세 조정 [3]
- 도메인 시프트가 있을 때 효과적
- 다만 competition 환경에서 적용 어려움 (제출 시 추론만 가능)

**방법**: TTA(Test-Time Augmentation)를 극대화하는 것이 현실적 대안
```
현재: 3 scales × 2 flips = 6 variants
개선: 5 scales × 2 flips × 3 brightness = 30 variants
```

**예상 효과**: 소폭 개선 (5~10%)
**확신도**: 중

---

## 3. 권장 방향: 동시 적용 조합

### 즉시 (코드 변경 최소):
1. Dev-only temperature scaling
2. TTA 확대 (brightness variation 추가)

### 단기 (1~2일):
3. Multi-backbone ensemble (ConvNeXt-V2-Base 추가 → 2-backbone ensemble)
4. OOF 가중치 최적화

### 중기 (2~3일):
5. DANN domain adversarial training 추가
6. 3~4 backbone ensemble로 확대
7. Stacking (LightGBM meta-learner)

### 예상 목표:

| 단계 | 전략 | 예상 LB |
|------|------|---------|
| 현재 | Video KD + EfficientNetV2-S | 0.058 |
| +A | Dev calibration + TTA 확대 | 0.04~0.05 |
| +B | 2-backbone ensemble | 0.03~0.04 |
| +C | DANN + 4-backbone ensemble | 0.015~0.025 |
| +stacking | Meta-learner calibration | 0.01~0.015 |

---

## 4. 가장 효과적인 단일 전략

**Multi-Backbone Ensemble이 ROI가 가장 높다.**

이유:
1. 코드 구조는 이미 완성됨 (backbone 이름만 바꾸면 됨)
2. 도메인 시프트에 대한 robustness가 자연스럽게 증가
3. 경쟁 대회에서 가장 보편적으로 검증된 전략
4. Video Teacher KD soft labels는 모든 backbone에 재사용 가능

**구체적 액션**:
1. `05_video_teacher_kd.ipynb`의 backbone만 바꿔서 3번 더 실행
2. 4개 backbone의 test predictions를 OOF 기반 가중 평균
3. Dev-only calibration 적용

---

## Sources

1. [Image Classification Tips from 13 Kaggle Competitions](https://neptune.ai/blog/image-classification-tips-and-tricks-from-13-kaggle-competitions)
2. [Domain-Adversarial Training of Neural Networks (Ganin et al.)](https://arxiv.org/abs/1505.07818)
3. [Test-Time Training Overview](https://www.emergentmind.com/topics/training-time-test)
4. [Ensemble Methods in Kaggle](https://www.toptal.com/developers/machine-learning/ensemble-methods-kaggle-machine-learn)
5. [Teacher Calibration in KD](https://arxiv.org/abs/2508.20224)
6. [DACON 구조물 안정성 대회 코드공유](https://dacon.io/en/competitions/official/236686/codeshare)
7. [DACON 대회 토론게시판](https://dacon.io/en/competitions/official/236686/talkboard)
8. [Probability Calibration (scikit-learn)](https://scikit-learn.org/stable/modules/calibration.html)
9. [Average Ensemble Optimization](https://guillaume-martin.github.io/average-ensemble-optimization.html)

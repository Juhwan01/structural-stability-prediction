# Research: v6 Augmentation 강화 + 앙상블 전략 (근거 기반)

## Executive Summary

v4 (LB 0.058) → 0.05 이하를 목표로, 3가지 전략을 근거 기반으로 설계.
모든 전략은 대회 규칙 내에서 실행 가능하며, test 데이터 학습 활용 금지 준수.

---

## 전략 1: 조명/카메라 Augmentation 강화

### 근거

대회 설명에서 domain shift의 정확한 원인을 명시:
- Train: "광원 및 카메라 좌표가 **고정**된 실험실 환경"
- Dev/Test: "광원 및 카메라 좌표가 **무작위로 변동**하는 실제 환경"

따라서 augmentation은 이 두 가지 변동을 **직접 시뮬레이션**해야 함.

### 현재 v4 augmentation 분석

```python
# 현재 (v4)
A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.2), contrast_limit=(-0.35, 0.35), p=0.8)
A.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.20, hue=0.04, p=0.6)
A.Perspective(scale=(0.02, 0.10), p=0.35)
A.Affine(scale=(0.92, 1.08), translate_percent=(-0.05, 0.05), rotate=(-7, 7), p=0.5)
```

### 문제점

1. **카메라 좌표 변동 시뮬레이션 부족**:
   - Perspective 0.02~0.10은 미세한 변형만 생성
   - 카메라 좌표가 "무작위"라면 훨씬 강한 시점 변화 가능
   - 연구: "Perspective transformation can produce new images captured from any camera's viewpoints" [IEEE, 2019]

2. **광원 변동 시뮬레이션 부족**:
   - Brightness/Contrast만으로는 비선형 조명 변화를 시뮬레이션 못함
   - Gamma correction이 빠져 있음 (카메라/디스플레이의 비선형 반응 시뮬레이션)
   - 그림자 효과가 빠져 있음 (광원 위치 변화 → 그림자 패턴 변화)
   - 연구: "accuracy drops 12% under gamma=60, adding RandomGamma(40,80) reduces the drop to 3%" [Albumentations docs]

3. **Hue 범위 너무 좁음**:
   - hue=0.04는 거의 변화 없음
   - 광원 색온도 변화를 시뮬레이션하려면 최소 0.08~0.12

### 제안 변경

```python
# 제안 (v6)
# === 광원 변동 시뮬레이션 ===
A.RandomGamma(gamma_limit=(60, 140), p=0.5),           # 비선형 조명 변화
A.RandomBrightnessContrast(
    brightness_limit=(-0.4, 0.3),                       # 약간 확대
    contrast_limit=(-0.4, 0.4), p=0.8),
A.ColorJitter(
    brightness=0.4, contrast=0.4,
    saturation=0.25, hue=0.10, p=0.6),                  # hue 0.04→0.10
A.RandomShadow(
    shadow_roi=(0, 0, 1, 1),
    num_shadows_limit=(1, 3),
    shadow_dimension=5, p=0.3),                          # 광원 위치 변화 시뮬레이션

# === 카메라 좌표 변동 시뮬레이션 ===
A.Perspective(scale=(0.03, 0.15), p=0.4),               # 0.10→0.15 확대
A.Affine(
    scale=(0.88, 1.12),                                  # 줌 범위 확대
    translate_percent=(-0.08, 0.08),                     # 이동 범위 확대
    rotate=(-12, 12), p=0.5),                            # ±7→±12
```

### 예상 효과: 중~높음 (confidence: 75%)
- 도메인 시프트의 원인을 직접 시뮬레이션
- 코드 변경 최소, 리스크 낮음
- augmentation이 너무 강하면 underfitting → 점진적 조정 필요

---

## 전략 2: 같은 Architecture 다중 Seed 앙상블

### 근거

- "Neural networks can have many optima, so that every run of training can potentially lead to a different solution. With standard CIFAR-10 configurations, there exist random seeds which differ by 1.3% in test-set accuracy" [arXiv:2304.01910]
- "Taking the average of multiple networks reduces the variance, as deep ANNs have high variance and low bias" [Ensemble Learning Methods, MachineLearningMastery]
- v5에서 다른 backbone 앙상블 시도 → 불안정한 backbone이 전체 오염 (실패 경험)
- 같은 architecture의 다중 seed는 가장 안전한 앙상블 방법

### 제안

```python
seeds = [42, 123, 777]  # 3개 seed
# 각 seed별로 v4 파이프라인 전체 실행
# 최종 예측 = 3개 모델의 probability 평균
```

### 예상 효과: 중간 (confidence: 80%)
- 안정적으로 variance 감소
- 0.005~0.01 LogLoss 개선 기대
- 단점: 학습 시간 3배

---

## 전략 3: Dev 2-Stage Fine-Tuning

### 근거

- "Few-shot Fine-tuning is All You Need for Source-free Domain Adaptation" [arXiv:2304.00792, ICLR 논문]
  - 핵심 주장: source pretrained model을 1~3 shot으로 fine-tune하는 것만으로 복잡한 DA 기법보다 실용적이고 신뢰도 높음
  - "carefully fine-tuned models do not suffer from overfitting even when trained with only a few labeled data"
- Dev 100장 = target domain (test와 동일 환경) 데이터
- 대회 규칙: "참가자는 학습 데이터(train)와 개발 데이터(dev)를 모델 학습에 모두 활용할 수 있습니다"

### 제안

```
Stage 1: Train 1000장 + Dev 100장으로 5-fold CV 학습 (현재 v4)
Stage 2: Best model에서 Dev 100장만으로 추가 fine-tune
         - lr: 1/10 (2e-5)
         - epochs: 3~5
         - strong regularization (dropout 높이기, weight decay 높이기)
         - early stopping on dev loss
```

### 주의점
- Dev를 Stage 2에서 학습에 쓰면, CV 검증이 불가능 (dev가 학습 데이터가 되므로)
- 따라서 Stage 1의 CV 결과를 기준으로 모델 선택 후, Stage 2는 "믿고 적용"
- 또는: Stage 1에서 train만 학습, dev는 validation으로만 → Stage 2에서 dev fine-tune

### 예상 효과: 중~높음 (confidence: 60%)
- 논문 근거가 있지만, 100장이 이 태스크에 충분한지 불확실
- overfitting 리스크 존재 → regularization이 핵심

---

## 실행 계획

### Phase 1: Augmentation 강화 (v6a)
- v4 코드에서 augmentation만 변경
- 동일한 effv2s backbone, KD, EMA, per-fold temp 유지
- 예상 시간: 코드 수정 10분 + 학습 40분

### Phase 2: 결과 확인 후 다중 Seed (v6b)
- v6a가 개선되면, 같은 설정으로 seed 3개 학습
- 예상 시간: 학습 2시간

### Phase 3: Dev Fine-Tuning (v6c)
- v6b의 best 모델에서 dev fine-tune
- 예상 시간: 학습 10분

---

## Sources

- [Perspective Transformation Data Augmentation for Object Detection (IEEE, 2019)](https://ieeexplore.ieee.org/abstract/document/8943416/)
- [RandomGamma - Albumentations](https://explore.albumentations.ai/transform/RandomGamma)
- [RandomShadow - Albumentations](https://explore.albumentations.ai/transform/RandomShadow)
- [Choosing Augmentations for Model Generalization - Albumentations](https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/)
- [Variance Between Runs of Neural Network Training (arXiv:2304.01910)](https://arxiv.org/pdf/2304.01910)
- [Ensemble Methods for Deep Learning Neural Networks](https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/)
- [Few-shot Fine-tuning is All You Need for Source-free Domain Adaptation (arXiv:2304.00792)](https://arxiv.org/abs/2304.00792)
- [Image Classification: Tips from 13 Kaggle Competitions](https://neptune.ai/blog/image-classification-tips-and-tricks-from-13-kaggle-competitions)
- [Data Augmentation: Ultimate Guide 2025 - Ultralytics](https://www.ultralytics.com/blog/the-ultimate-guide-to-data-augmentation-in-2025)

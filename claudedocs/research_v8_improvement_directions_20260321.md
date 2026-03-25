# Research: v8 (LB 0.0385) 개선 방향

## 현재 상태

| Version | CV | LB | 비고 |
|---------|------|------|------|
| v4 (effv2s) | 0.0429 | 0.058 | baseline |
| v8 (convnextv2) | 0.0301 | **0.0385** | 현재 최고 LB |
| v9 (label smooth+focal+mixup) | 0.0138 | v8보다 나쁨 | CV↑ LB↓ = overfitting |
| **1위** | - | **0.00782** | 목표 |

## v9 실패 원인 분석

v9는 CV 0.0138로 좋아 보였지만 LB에서 악화. 원인:
1. **Label Smoothing + KD soft label 충돌**: teacher soft label에 smoothing을 또 적용하면 정보 손실
2. **Focal Loss가 LogLoss calibration 왜곡**: focal weighting이 확률값 calibration을 방해
3. **lr을 너무 낮춘 것 (5e-5)**: overfitting 구간에서 오래 머물러 train domain에 과적합
4. **4가지를 동시에 바꿔서** 어떤 게 좋고 나쁜지 분리 불가

**교훈**: CV-LB 갭이 큰 도메인 시프트 문제에서는 CV 개선 ≠ LB 개선. regularization을 추가하면 train domain 내에서는 좋지만 domain gap은 못 줄임.

## v8 기반 개선 방향 (우선순위)

### 1. Domain-Invariant Feature 학습 (가장 핵심)

현재 bottleneck은 domain shift. CV를 줄이는 게 아니라 **CV-LB 갭을 줄이는** 방향이어야 함.

**방법 A: Adversarial Domain Adaptation (DANN)**
- domain discriminator를 붙여서 train/dev feature를 구분 못 하게 학습
- dev 100장이 target domain 대표 → adversarial로 domain-invariant feature 유도
- 구현: GradientReversalLayer + domain classifier head 추가
- 참고: [Adversarial Domain Adaptation for Cross-Domain Classification](https://www.nature.com/articles/s41598-025-95390-3)

**방법 B: Style Transfer (AdaIN) augmentation**
- dev 이미지의 feature statistics(mean, std)를 train 이미지에 적용
- train 이미지가 dev/test와 비슷한 스타일로 변환됨
- test 통계를 안 쓰므로 규칙 준수
- 구현: AdaIN layer, dev 100장의 feature statistics 추출

**방법 C: Frequency-based domain alignment**
- train과 dev/test의 주파수 스펙트럼 차이(저품질 vs 고품질)를 분석
- FDA(Fourier Domain Adaptation): train 이미지의 저주파를 dev 이미지의 저주파로 교체
- 규칙 준수 (dev 통계만 사용)

### 2. 더 나은 앙상블 전략

**방법 D: 다양한 backbone 앙상블**
- v4 (effv2s) + v8 (convnextv2) 이미 있음 → 가중 앙상블
- OOF 기반 최적 가중치: v4=0.30, v8=0.70 → OOF 0.0196
- 추가로 EVA-02 or SwinV2로 3번째 모델 학습하면 다양성 증가
- **LB 제출 한번 안 한 상태**: v4+v8 앙상블 submission은 이미 생성됨 (`v4_v8_ensemble_submission.csv`)

**방법 E: Stacking**
- OOF predictions을 feature로 사용하여 2nd level model (LR, Ridge) 학습
- calibration이 자동으로 이루어짐

### 3. Calibration 개선 (Quick Win)

**방법 F: Global Temperature Scaling 대신 Platt Scaling**
- 현재: per-fold temperature scaling (fold마다 다른 temperature)
- 개선: dev set으로만 temperature를 fitting (target domain calibration)
- dev가 test와 같은 도메인이므로 더 정확한 calibration

**방법 G: Isotonic Regression calibration**
- temperature scaling보다 유연한 비모수적 calibration
- LogLoss에 직접적으로 효과적

### 4. 학습 전략 개선

**방법 H: 2-Stage Training**
- Stage 1: train 1000장으로 pretraining (현재와 동일)
- Stage 2: dev 100장으로 low-lr fine-tuning (target domain adaptation)
- dev가 test와 같은 도메인이므로 domain-specific feature 학습

**방법 I: Repeated K-Fold with different seeds**
- seed를 바꿔서 여러 번 학습 → 예측 평균
- 단순하지만 분산 감소에 효과적

## 추천 실행 순서

1. **즉시 (5분)**: v4+v8 앙상블 제출 (이미 생성됨) → LB 확인
2. **다음 (1시간)**: Style Transfer (AdaIN) augmentation 추가 → v8 재학습
3. **그 다음 (2시간)**: Adversarial Domain Adaptation 구현
4. **병렬**: EVA-02로 3번째 모델 학습 → 3-model 앙상블

## 핵심 인사이트

> **CV를 낮추는 것이 아니라, domain gap을 줄이는 것이 목표.**
> v9처럼 regularization을 추가해도 train domain 내에서만 좋아지고 test에서는 역효과.
> domain gap을 직접 공략하는 방법(Style Transfer, DANN, FDA)이 필요.

## Sources

- [Survey of Data Augmentation in Domain Generalization (2025)](https://link.springer.com/article/10.1007/s11063-025-11747-9)
- [Adversarial Domain Adaptation for Classification (2025)](https://www.nature.com/articles/s41598-025-95390-3)
- [Test-Time Modification: Inverse Domain Transform (2024)](https://arxiv.org/html/2512.13454)
- [CV-LB Gap Handling in Kaggle](https://www.kaggle.com/discussions/questions-and-answers/504457)
- [Kaggle Handbook: Surviving Shake-up](https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-fundamentals-to-survive-a-kaggle-shake-up-3dec0c085bc8)

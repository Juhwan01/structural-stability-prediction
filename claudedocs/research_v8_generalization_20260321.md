# Research: v8 일반화 개선 방향 (대회 목적 기반)

## 대회 핵심 목적

> "다양한 물리적 요소(무게중심, 층별 하중 분포, 구조 배치 패턴)를 종합적으로 고려한 **정밀 분석**"

**일반화**가 핵심. train 도메인에 과적합하지 않고 unseen test 도메인에서도 잘 동작해야 함.

## 현재 상태

| Version | CV | LB | 비고 |
|---------|------|------|------|
| v8 (ConvNeXt-V2-Base) | 0.0242 | **0.0385** | 현재 최고 LB |
| v9 (regularization 추가) | 0.0138 | 악화 | CV↑ LB↓ |
| v10 (FDA) | 0.0139 | 미제출 (v9 패턴) | CV↑ LB↓ 예상 |
| **1위** | - | **0.00782** | 목표 |

**핵심 문제**: CV 개선이 LB로 전혀 이어지지 않음. 근본적으로 다른 접근이 필요.

## 방향 1: VLM Backbone (SigLIP) — 가장 유망

### 근거
- ConvNeXt-V2는 ImageNet supervised/self-supervised pretrain → 특정 도메인 bias가 강함
- **SigLIP/CLIP**은 4억+ image-text 쌍으로 pretrain → 도메인에 구애받지 않는 visual representation
- "조명이 다르다", "해상도가 다르다" 같은 low-level 차이에 훨씬 robust
- timm에서 바로 사용 가능: `vit_base_patch16_siglip_384.v2_webli` (93M params, 384px)
- 기존 파이프라인에서 backbone만 교체하면 됨

### 구현
```python
backbone: str = 'vit_base_patch16_siglip_384.v2_webli'
img_size: int = 384
batch_size: int = 4  # ConvNeXt-V2와 비슷한 크기
lr: float = 5e-5  # ViT는 낮은 lr
```

### 기대 효과
- domain-invariant feature → CV-LB 갭 자체가 줄어듦
- 1위 솔루션이 VLM backbone을 사용했을 가능성 높음

### Sources
- [SigLIP 2: Multilingual Vision-Language Encoders (2025)](https://arxiv.org/pdf/2502.14786)
- [Fine-Tuning SigLIP2 for Image Classification](https://huggingface.co/blog/prithivMLmods/siglip2-finetune-image-classification)
- [Leveraging VLMs for Domain Generalization (CVPR 2024)](https://arxiv.org/abs/2310.08255)

---

## 방향 2: Dev-only Temperature Scaling — 즉시 적용 가능

### 근거
- 현재 per-fold temp은 train+dev 혼합 validation에서 fitting
- test 도메인 = dev 도메인이므로, **dev 100장만으로 temp을 다시 calibration**하면 LogLoss 직접 개선
- 재학습 불필요, v8 기존 모델에 바로 적용

### 구현
```python
# 각 fold 모델로 dev 100장 예측 → dev-only temperature fitting
dev_logits = model.predict(dev_data)
dev_temp = TemperatureScaler().fit(dev_logits, dev_labels)
# test 예측에 dev_temp 적용
test_probs = sigmoid(test_logits / dev_temp)
```

### 기대 효과
- LogLoss에 직접 영향 (calibration 개선)
- 소폭 개선 예상 (0.001~0.005)

### Sources
- [Calibration of Network Confidence for Unsupervised Domain Adaptation](https://arxiv.org/html/2409.04241)

---

## 방향 3: 2-Stage Dev Fine-tune — 중간 난이도

### 근거
- Stage 1: 전체 데이터(train+dev)로 학습 (현재 v8)
- Stage 2: **backbone 동결 + dev 100장으로 head만 미세 조정** (lr=1e-6, 2-3 epochs)
- dev = test 도메인이므로, head가 target 도메인에 맞춰짐
- KDD 2024 "Practical Single Domain Generalization" 논문에서도 training-time + test-time 학습 조합 제안

### 구현
- v8 학습 후, 각 fold 모델에 dev fine-tune stage 추가
- backbone freeze, classifier/fusion만 학습
- hard label만 사용 (teacher soft label은 train 도메인이므로 제외)

### 기대 효과
- target domain에 head를 맞추므로 LB 개선 가능
- 단, dev 100장이 적어서 overfitting 위험 → 극히 낮은 lr 필수

### Sources
- [Practical Single Domain Generalization (KDD 2024)](https://dl.acm.org/doi/10.1145/3637528.3671806)

---

## 방향 4: Test-Time Normalization Adaptation — 재학습 불필요

### 근거
- ConvNeXt-V2는 LayerNorm 사용 → 전통적 BN adaptation은 효과 제한적
- 대신 **TENT (entropy minimization)**: test 데이터의 예측 entropy를 줄이도록 LN affine params(gamma, beta)만 미세 조정
- 재학습 없이 inference 시 적용

### 주의
- ConvNeXt-V2는 BatchNorm이 아닌 LayerNorm → TENT는 동작하지만 BN adaptation은 효과 없음
- 1 step만 사용, lr 매우 작게 (1e-5)

### Sources
- [TENT: Fully Test-Time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726)
- [Beyond Model Adaptation at Test Time: A Survey](https://arxiv.org/html/2411.03687v1)

---

## 추천 실행 순서

| 순서 | 방법 | 재학습 | 소요 | 기대 |
|------|------|--------|------|------|
| **1** | SigLIP backbone (방향1) | O (전체) | 2시간 | **HIGH** — 근본적 접근 |
| **2** | Dev-only temp scaling (방향2) | X | 15분 | LOW-MED — quick win |
| **3** | 2-Stage dev fine-tune (방향3) | 부분 | 30분 | MED |
| **4** | TENT (방향4) | X | 30분 | LOW-MED |

**핵심 인사이트**: regularization이나 augmentation 트릭은 다 실패했음. **backbone 자체를 domain-robust한 것으로 바꾸는 게 가장 근본적**. SigLIP은 400M+ 다양한 데이터로 pretrain되어 이미지 품질/조명/카메라 차이에 inherently robust함.

# V3 학습 코드 타당성 검증 리서치 리포트

**날짜**: 2026-03-17
**목적**: v3 학습 코드의 기법별 타당성 검증 + 성능 최대화 관점 개선안
**평가 기준**: LogLoss (결과 품질 최우선, 속도 무관)

---

## Executive Summary

v3 코드는 개별 기법들은 대부분 학술적으로 타당하나, **조합과 튜닝에서 심각한 문제**가 있다. v1(ConvNeXt-V2-Base) LogLoss **0.1296** → v3(EfficientNetV2-S) LogLoss **2.4077**로 **18.6배 악화**. 이는 단순 백본 교체가 아닌 구조적 문제를 시사한다.

---

## 1. 버전별 성능 비교

| Version | Backbone | CV LogLoss | 평가 |
|---------|----------|-----------|------|
| v1 | ConvNeXt-V2-Base | **0.1296** | 우수 |
| v2 | ConvNeXt-V2-Base (수정) | 0.3685 | 악화 |
| v3 | EfficientNetV2-S | 2.4077 | 심각 |

**핵심 관찰**: v1→v2→v3로 갈수록 기법은 복잡해지는데 성능은 지속 하락. "더 많은 기법 = 더 나은 성능"이 아님을 명확히 보여줌.

---

## 2. 기법별 타당성 분석

### 2.1 백본 선택: EfficientNetV2-S ⚠️ 문제 있음

**현재**: `tf_efficientnetv2_s.in21k_ft_in1k` (320px, 49.9M params)
**v1**: ConvNeXt-V2-Base (더 큰 모델)

**연구 결과**:
- 소규모 데이터셋에서 **ConvNeXt가 EfficientNetV2보다 fine-tuning 성능 우월** [1]
- EfficientNetV2-S는 ImageNet 정확도가 높지만, **도메인 특화 fine-tuning에서는 최고가 아님** [1]
- 순수 CNN이 Transformer보다 소규모 데이터 fine-tuning에서 우위 [1]
- **1,100개 샘플**은 명백히 소규모 → ConvNeXt-V2-Base가 더 적합

**판정**: ❌ **백본 다운그레이드**. v1의 ConvNeXt-V2-Base를 유지하는 것이 합리적.

### 2.2 Dual-Stream + Transformer Fusion ⚠️ 과도한 복잡성

**현재**: 2-layer 8-head Transformer encoder + view embeddings + concat

**연구 결과**:
- Multi-view fusion에서 Transformer는 **충분한 데이터**가 있을 때 효과적 [2]
- 1,100개 샘플에서 8-head attention은 **과적합 위험** 높음
- WACV 2024 연구: late fusion + mutual distillation이 소규모에서 더 안정적 [3]
- v1의 단순 concat이 1,100개에서는 오히려 더 적합할 수 있음

**판정**: ⚠️ Transformer fusion 자체는 타당하나, **데이터 규모 대비 과도**. 1-layer 4-head로 축소하거나, 단순 SE-attention 또는 concat으로 대체 권장.

### 2.3 Physics-Informed Motion Targets ✅ 아이디어는 좋음, ⚠️ 구현에 문제

**현재**:
- 비디오에서 MAD(Mean Absolute Difference) 추출
- Soft target: `0.65 * min(max_diff/10, 1.5) + 0.35 * min(mean_diff/0.15, 1.5)`
- Unstable: 0.65~0.98, Stable: 0.02~0.15

**타당성 분석**:
- 물리 시뮬레이션 정보를 supervision signal로 활용하는 것은 **학술적으로 우수한 접근** [4]
- 그러나 **soft target 범위가 너무 극단적**: stable이 최대 0.15, unstable이 최소 0.65
- LogLoss에서 이런 범위 제한은 **calibration 왜곡**을 유발
- BCEWithLogits + soft target 조합에서 모델이 0.15~0.65 사이를 예측하면 **양쪽 모두에서 큰 loss 발생**
- 이것이 **v3 LogLoss 2.4077의 주요 원인일 가능성 높음**

**판정**: ⚠️ soft target의 범위 설계가 LogLoss 최적화에 역행. Hard label (0/1) 또는 매우 약한 smoothing (0.02~0.98)으로 변경 필요.

### 2.4 Multi-Task Auxiliary Losses ⚠️ 가중치 문제

**현재**: main(1.0) + motion(0.20) + onset(0.15) + severity(0.15) = 총 1.50

**연구 결과**:
- Kendall et al. (CVPR 2018): 수동 가중치 대신 **homoscedastic uncertainty 기반 자동 가중치**가 우월 [5]
- 보조 손실 총합이 주 손실의 50%에 달하면 **주 태스크 학습 방해** 가능
- 특히 1,100개 소규모 데이터에서 4개 태스크 동시 학습은 **과도**

**판정**: ⚠️ 보조 손실 가중치를 0.05~0.10 수준으로 대폭 축소하거나, uncertainty-based weighting 적용 필요.

### 2.5 GeM Pooling ✅ 타당

**현재**: GeM(p=3.0, learnable)

**연구 결과**:
- Image retrieval에서 검증된 기법 [6]
- Classification에서는 marginal gain이지만 harm도 없음
- Group GeM (GGeM)이 ViT에서 0.1~0.7%p 향상 [6]

**판정**: ✅ 사용해도 무방하지만, 성능 차이의 주범은 아님.

### 2.6 Augmentation Pipeline ✅ 대체로 타당, 일부 과도

**현재**: CLAHE + ColorJitter + Perspective + Affine + Mixup/CutMix + CoarseDropout 등

**분석**:
- **도메인 시프트 대응에 필수적**: train(고정 조명/카메라) vs test(랜덤 조명/카메라) [7]
- 색상/밝기 augmentation이 특히 중요 (조명 변화 대응)
- Perspective/Affine는 카메라 각도 변화 대응에 적합
- 그러나 **CoarseDropout(4~8 holes, 16~48px)**은 320px에서 상당히 공격적
- Mixup Beta(0.3, 0.3)은 U-shaped distribution → 극단적 혼합 비율이 빈번

**판정**: ✅ 방향은 맞지만, CoarseDropout 축소(2~4 holes)와 Mixup alpha 증가(0.4~0.5)로 조정 권장.

### 2.7 Cross-Validation: StratifiedGroupKFold ✅ 우수

**현재**: label + source_domain으로 stratify, geometry cluster로 group

**분석**:
- 구조물 유사성 기반 그룹 분할은 **leakage 방지에 매우 효과적**
- 16개 geometry cluster는 1,100개 샘플에 적절한 granularity
- dev 100개를 포함하여 모든 fold에서 domain shift를 학습하는 전략도 타당

**판정**: ✅ 이 부분은 v3에서 가장 잘 설계된 요소.

### 2.8 EMA + SWA ✅ 타당

**현재**: EMA(decay=0.9995) + SWA(start=75%, lr=1e-5)

**연구 결과**:
- EMA: 소규모 데이터에서 implicit regularization 효과, noisy label에 robust [8]
- SWA: wider optima로 수렴, train/test error surface gap 감소 [8]
- Adaptive SWA (ASWA): val 성능 개선 시에만 averaging하여 소규모에서 더 효과적 [8]

**판정**: ✅ 두 기법 모두 소규모 데이터셋에서 검증된 정규화 전략. 현재 구현 적절.

### 2.9 Temperature Scaling ✅ 기법은 타당, ⚠️ soft target과 충돌

**현재**: LBFGS 기반 per-fold temperature scaling

**분석**:
- Temperature scaling 자체는 calibration의 표준 기법 [9]
- 그러나 **soft target으로 학습한 모델에 temperature scaling 적용**은 이중 보정
- Soft target이 이미 calibration 역할을 하므로, 추가 temperature scaling이 **과보정**할 수 있음
- 특히 validation 셋이 작을 때(~220개/fold) temperature overfitting 위험

**판정**: ⚠️ soft target 사용 시 temperature scaling은 불필요하거나 역효과. hard label 사용 시에만 적용.

### 2.10 TTA (Test-Time Augmentation) ✅ 효과적

**현재**: 3 scales(320, 384, 448) × 2 flips = 6 variants

**분석**:
- Multi-scale TTA는 robustness 향상에 효과적
- 도메인 시프트가 있는 대회에서 특히 유용
- 6 variants는 적절한 수준

**판정**: ✅ 잘 설계됨.

---

## 3. 핵심 문제 진단: v3 LogLoss 2.4077의 원인

### 원인 1 (주요): Soft Target 범위의 LogLoss 왜곡

```
- Stable 샘플의 soft target: 0.02 ~ 0.15
- Unstable 샘플의 soft target: 0.65 ~ 0.98
- 모델이 이 범위에 맞춰 학습 → 실제 평가는 hard label(0/1)로 수행
- 모델이 stable에 대해 최대 0.15 확률을 예측하도록 학습됨
- 실제 label=0에 대해 모델이 0.15를 예측하면 LogLoss = -log(1-0.15) = 0.163
- label=0에 대해 0.01을 예측하면 LogLoss = -log(1-0.01) = 0.01
- → soft target이 모델의 확신도를 인위적으로 제한하여 LogLoss 악화
```

### 원인 2: 백본 다운그레이드
- ConvNeXt-V2-Base → EfficientNetV2-S는 소규모 데이터에서 성능 하락 요인

### 원인 3: Multi-Task Loss의 주 태스크 방해
- 보조 손실 합계가 주 손실의 50%로 너무 높음
- 1,100개에서 4개 태스크 동시 학습은 과적합 + 주 태스크 학습 방해

### 원인 4: Transformer Fusion 과적합
- 2-layer 8-head attention은 1,100개 데이터에서 과도한 용량

---

## 4. 성능 최대화를 위한 구체적 권장사항

### 우선순위 1 (Critical): Soft Target 제거 또는 수정

```python
# 방법 A: Hard label로 복귀 (가장 안전)
target = label_int  # 0 or 1

# 방법 B: 매우 약한 label smoothing (0.01)
target = label_int * (1 - 0.02) + 0.01  # stable:0.01, unstable:0.99
```

### 우선순위 2 (High): 백본을 ConvNeXt-V2-Base로 복원

```python
backbone = 'convnextv2_base.fcmae_ft_in22k_in1k'  # v1에서 사용한 모델
img_size = 384  # ConvNeXt 표준 해상도
```

### 우선순위 3 (High): Multi-Task Loss 가중치 대폭 축소

```python
# 현재: motion=0.20, onset=0.15, severity=0.15 (합계 0.50)
# 권장: motion=0.05, onset=0.03, severity=0.03 (합계 0.11)
# 또는 uncertainty-based auto-weighting 적용
```

### 우선순위 4 (Medium): Fusion 단순화

```python
# 현재: 2-layer 8-head Transformer
# 권장: 단순 concat + SE-attention 또는 1-layer 4-head
# v1의 단순 concat이 1,100개에서 더 안정적
```

### 우선순위 5 (Medium): Domain Shift 대응 강화

이 대회의 핵심 난이도는 **train(고정 환경) vs test(랜덤 환경)**의 domain shift.

```python
# 색상/밝기 augmentation 강화 (이미 적절하지만 더 강화 가능)
RandomBrightnessContrast(brightness=[-0.4, 0.2], contrast=[-0.4, 0.4], p=0.9)
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.7)
# 배경색 변화 augmentation 추가 고려
```

### 우선순위 6 (Low): 앙상블 전략

최종 성능 극대화를 위해:
```
- ConvNeXt-V2-Base (v1 스타일) × 5 folds
- EfficientNetV2-S (수정된 v3) × 5 folds
- SwinV2-Small/Base × 5 folds
- 총 15개 모델의 가중 평균 (OOF LogLoss 기반 가중치 최적화)
```

---

## 5. v3에서 유지할 가치가 있는 기법들

| 기법 | 유지? | 이유 |
|------|-------|------|
| Geometry-clustered fold splitting | ✅ | Leakage 방지에 매우 효과적 |
| EMA + SWA | ✅ | 소규모 데이터 정규화에 검증됨 |
| Center physics crop | ✅ | 불필요한 배경 제거 |
| Checkerboard rotation normalization | ✅ | Top view 정규화에 유용 |
| Multi-scale TTA | ✅ | Robustness 향상 |
| Gradient accumulation (effective batch 32) | ✅ | 소규모에서 안정적 학습 |
| Motion target 추출 | ⚠️ | 아이디어는 좋지만 적용 방식 수정 필요 |

---

## 6. 권장 실험 순서

1. **v1 코드 기반으로 v3의 좋은 기법만 이식** (geometry fold, center crop, checkerboard norm, EMA)
2. **Soft target 제거하고 hard label + 약한 label smoothing(0.01)으로 교체**
3. **Multi-task loss 제거하거나 가중치 0.05 이하로 축소**
4. **ConvNeXt-V2-Base 백본 유지, 이미지 384px**
5. 위 조합으로 baseline CV 측정 → v1(0.1296) 이하 목표
6. 점진적으로 Transformer fusion, motion soft target 등 추가하며 ablation

---

## Sources

1. [Resource-efficient Domain Specific Backbone Comparison](https://arxiv.org/html/2406.05612v1) - 소규모 데이터셋에서 ConvNeXt가 EfficientNetV2보다 fine-tuning 성능 우월
2. [TGF-Net: Transformer and CNN Fusion](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0316900) - 멀티모달 Transformer fusion 전략
3. [Multi-View Classification Using Hybrid Fusion (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/papers/Black_Multi-View_Classification_Using_Hybrid_Fusion_and_Mutual_Distillation_WACV_2024_paper.pdf) - 소규모 multi-view에서 late fusion + distillation
4. [Multi-Objective Loss Balancing for Physics-Informed DL](https://www.sciencedirect.com/science/article/pii/S0045782525001860) - 물리 기반 다목적 손실 밸런싱
5. [Multi-Task Learning Using Uncertainty (Kendall, CVPR 2018)](https://arxiv.org/abs/1705.07115) - Homoscedastic uncertainty 기반 자동 가중치
6. [Group GeM Pooling for ViT](https://arxiv.org/abs/2212.04114) - GeM pooling 효과
7. [Domain Shift Practical Guide](https://www.numberanalytics.com/blog/domain-shift-deep-learning-practical-guide) - 도메인 시프트 대응 전략
8. [EMA Dynamics and Benefits (2024)](https://arxiv.org/html/2411.18704v1) - EMA/SWA 소규모 데이터 효과
9. [Temperature Scaling (gpleiss)](https://github.com/gpleiss/temperature_scaling) - 표준 calibration 구현
10. [Label Smoothing Degrades Selective Classification (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/9dc5accb1e4f4a9798eae145f2e4869b-Paper-Conference.pdf) - Label smoothing의 calibration 왜곡
11. [DACON 구조물 안정성 물리 추론 AI 경진대회](https://dacon.io/en/competitions/official/236686/overview/description) - 대회 개요 및 평가 기준

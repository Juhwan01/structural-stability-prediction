# Research: v10 실패 후 전략 재수립 — v8 기반 SOTA 추출 방안

> Date: 2026-03-22
> Depth: Deep
> Confidence: HIGH (이전 실험 데이터 + 문헌 + 대회 분석 기반)

---

## Executive Summary

v10(SigLIP)은 Fold 2 폭발(LogLoss 4.08)로 실패. v9(과도한 regularization)도 LB 악화.
**v8(ConvNeXt-V2, CV 0.0242, LB 0.0385)이 여전히 최고 LB**.
1위(LB 0.00782) 대비 ~5배 차이. 이 갭을 줄이기 위한 실행 가능한 전략을 정리.

**핵심 결론**: backbone 교체(v10)보다 **domain gap 해소 + 앙상블 다양성 + calibration 정밀화**가 더 효과적.

---

## 1. 현재 상태 요약

| Version | Backbone | CV LogLoss | LB LogLoss | 비고 |
|---------|----------|------------|------------|------|
| v4 | EfficientNetV2-S | 0.0429 | 0.058 | baseline |
| v8 | ConvNeXt-V2-Base | 0.0242 | **0.0385** | 현재 최고 LB |
| v9 | ConvNeXt-V2 + focal+smooth | 0.0000 | v8보다 나쁨 | CV 과적합 |
| v10 | SigLIP ViT-B/16-384 | 1.9616 | 미제출 | Fold 2 폭발 |
| **1위** | - | - | **0.00782** | 목표 |
| v4+v8 앙상블 | - | 0.0196 | 미확인 | 아직 제출 안 함 |

### v10 실패 원인

- **ViT는 소규모 데이터에 취약**: 1,100장에서 SigLIP ViT는 CNN 대비 일반화 부족
- Fold 2에서 학습 자체가 안 됨 (val_auc 0.55~0.86, 다른 fold는 0.99+)
- geometry fold 분할에서 특정 그룹이 Fold 2에 집중 → ViT가 해당 패턴 학습 실패
- **문헌 확인**: "low data fine-tune regime에서는 ViT 대신 **CNN(ConvNeXt, EfficientNet, RegNet)**이 우수" (Resource-efficient Domain Specific Comparison, 2024)

### 핵심 교훈

1. **CV 개선 ≠ LB 개선** (v9 교훈)
2. **backbone을 무리하게 바꾸면 불안정** (v10 교훈)
3. **domain gap이 진짜 bottleneck** (v8 → LB 갭 0.014)

---

## 2. 선택 가능한 방법론 (좁혀놓은 후보)

### Tier 1: 즉시 실행 가능, 높은 기대효과

#### A. 다중 backbone 앙상블 (Expected LB improvement: 0.005~0.010)

v8 코드에서 backbone만 교체하여 2~3개 추가 모델 학습 후 가중 앙상블.

**왜 효과적인가?**
- 캐글/데이콘 상위팀 공통점: 앙상블이 단일 모델 대비 항상 우위
- CNN끼리도 architecture가 다르면 오류 상관이 낮아 앙상블 효과 큼
- v4+v8 OOF 앙상블이 이미 0.0196으로 개선됨

**추천 backbone 조합**:

| Model | timm name | Params | 특징 |
|-------|-----------|--------|------|
| ConvNeXt-V2-Base (v8) | `convnextv2_base.fcmae_ft_in22k_in1k` | 89M | 현재 best |
| EfficientNetV2-S (v4) | `tf_efficientnetv2_s.in21k_ft_in1k` | 21M | 이미 있음 |
| **EVA-02-Small** | `eva02_small_patch14_336.mim_in22k_ft_in1k` | 22M | MIM pretrain, 강한 feature |
| **ConvNeXt-Base (CLIP)** | `convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384` | 89M | CLIP pretrain, domain robust |

**앙상블 방법**: OOF 기반 scipy.optimize 가중치 최적화 → weighted average

#### B. Domain Adaptation — FDA (Fourier Domain Adaptation) (Expected: 0.003~0.008)

train(시뮬레이션)과 dev/test(실사) 간 도메인 차이를 주파수 영역에서 줄임.

**원리**: train 이미지의 저주파 성분(전체 색상/밝기 분포)을 dev 이미지의 저주파로 교체.
**장점**: 구현 간단 (~30줄), 학습 과정 변경 없음 (augmentation에 추가만)
**주의**: v10에서 FDA submission 파일이 이미 존재 → 이전 시도가 있었을 수 있음

```python
def fda_transfer(src_img, trg_img, beta=0.01):
    """저주파 스펙트럼 교환으로 domain alignment"""
    src_fft = np.fft.fft2(src_img, axes=(0,1))
    trg_fft = np.fft.fft2(trg_img, axes=(0,1))
    h, w = src_img.shape[:2]
    b_h, b_w = int(h * beta), int(w * beta)
    # 중심부(저주파) 교체
    src_fft[:b_h, :b_w] = trg_fft[:b_h, :b_w]
    src_fft[-b_h:, -b_w:] = trg_fft[-b_h:, -b_w:]
    return np.real(np.fft.ifft2(src_fft, axes=(0,1))).clip(0, 255).astype(np.uint8)
```

#### C. Dev-based Temperature Scaling (Expected: 0.002~0.005)

현재: per-fold OOF로 temperature fitting.
개선: **dev 100장으로 temperature fitting** (dev = test domain이므로 더 정확한 calibration).

```python
# dev 예측 → 최적 temperature 탐색
from scipy.optimize import minimize_scalar
def optimal_temp(logits, labels):
    def loss_fn(t):
        probs = sigmoid(logits / t)
        return log_loss(labels, probs)
    result = minimize_scalar(loss_fn, bounds=(0.01, 10.0), method='bounded')
    return result.x
```

---

### Tier 2: 구현 노력 중간, 확실한 효과

#### D. 2-Stage Training (Expected: 0.003~0.007)

- Stage 1: train 1,000장으로 표준 학습 (현재 v8과 동일)
- Stage 2: dev 100장으로 low-lr fine-tuning (lr * 0.01, 3~5 epoch)
- dev가 test와 같은 도메인이므로 **target domain 적응 효과**

**주의**: dev 100장으로 overfitting 위험 → 강한 augmentation + early stopping 필수

#### E. Repeated K-Fold (다중 seed) (Expected: 0.002~0.004)

- seed 3개 (42, 123, 2024)로 각각 5-fold 학습
- 총 15개 모델의 예측 평균
- **분산 감소 + fold 불운 제거** (v10 Fold 2 같은 문제 방지)
- 단순하지만 안정적

#### F. Stacking Ensemble (Expected: 0.003~0.006)

- v4, v8, (+ EVA-02, ConvNeXt-CLIP) OOF predictions을 feature로
- 2nd level: Logistic Regression / Ridge → 자동 calibration
- **calibration 오류를 학습으로 보정**하므로 temperature scaling보다 유연

---

### Tier 3: 고급, 구현 복잡하지만 차별화 가능

#### G. Adversarial Domain Adaptation (DANN) (Expected: 0.005~0.010)

- Gradient Reversal Layer로 domain-invariant feature 학습
- train/dev 구분 못하는 feature → test에서도 일반화
- 구현 복잡도 높지만 domain shift 문제의 정석 해법

#### H. Test-Time Adaptation (TTA+) (Expected: 0.001~0.003)

- 표준 TTA (flip, multi-scale) + **Entropy Minimization TTA**
- test 이미지에 대해 모델 파라미터를 소량 조정 (batch norm stats 업데이트)
- 참고: Tent (Test-Time Entropy Minimization)

---

## 3. SOTA 달성을 위한 추천 실행 계획

### Phase 1: Quick Wins (1~2시간)

1. **v4+v8 앙상블 제출** → LB 확인 (이미 파일 있음)
2. **Dev-based Temperature Scaling** 적용 → v8 재제출

### Phase 2: 모델 다양성 확보 (4~6시간)

3. **EVA-02-Small로 v8 코드 재학습** (backbone만 교체)
4. **ConvNeXt-Base-CLIP으로 v8 코드 재학습**
5. **4-model 가중 앙상블** (v4 + v8 + EVA + CLIP)

### Phase 3: Domain Gap 공략 (2~3시간)

6. **FDA augmentation 추가** → v8 재학습
7. **2-Stage Training** (dev fine-tuning)

### Phase 4: 최종 조합 (1시간)

8. **Stacking 앙상블** (OOF → LR/Ridge 2nd level)
9. **최종 제출**

---

## 4. 방법론별 비교 매트릭스

| 방법 | 난이도 | 기대효과 | 리스크 | 우선순위 |
|------|--------|----------|--------|----------|
| A. 다중 backbone 앙상블 | 낮음 | ★★★★ | 낮음 | **1** |
| B. FDA augmentation | 낮음 | ★★★ | 중간 | **2** |
| C. Dev Temperature Scaling | 매우 낮음 | ★★ | 낮음 | **3** |
| D. 2-Stage Training | 중간 | ★★★ | 중간 | **4** |
| E. Repeated K-Fold | 낮음 (시간만) | ★★ | 낮음 | **5** |
| F. Stacking | 중간 | ★★★ | 낮음 | **6** |
| G. DANN | 높음 | ★★★★ | 높음 | 7 |
| H. TTA+ | 중간 | ★ | 중간 | 8 |

---

## 5. 예상 최종 성능

| 단계 | 예상 LB |
|------|---------|
| 현재 v8 단일 | 0.0385 |
| + v4 앙상블 | 0.030~0.033 |
| + EVA/CLIP 추가 앙상블 | 0.022~0.028 |
| + FDA + Dev calibration | 0.018~0.024 |
| + Stacking + 2-Stage | 0.013~0.020 |
| 1위 | 0.00782 |

> 현실적 목표: **LB 0.015~0.020** (상위 5~10%)

---

## 6. 하지 말아야 할 것

1. **ViT 계열 단독 사용 금지** — 1,100장에서 ViT는 CNN 대비 불안정 (v10 증명)
2. **v9처럼 여러 기법 동시 변경 금지** — 한 번에 하나씩 변경, LB로 검증
3. **CV만 보고 판단 금지** — CV 0.0000은 과적합 신호 (v9 교훈)
4. **Focal Loss 사용 금지** — LogLoss 평가 대회에서 calibration 왜곡

---

## Sources

- [Image Classification Tips from 13 Kaggle Competitions](https://neptune.ai/blog/image-classification-tips-and-tricks-from-13-kaggle-competitions)
- [Resource-efficient Domain Specific Comparison for CV (2024)](https://arxiv.org/html/2406.05612v1)
- [ViSwNeXtNet: Ensemble of ConvNeXt + Swin + ViT](https://www.mdpi.com/2075-4418/15/12/1507)
- [Ensemble of ConvNeXt V2 and MaxViT](https://arxiv.org/html/2410.10710v2)
- [ConvNeXt V2 Paper](https://www.researchgate.net/publication/373314265_ConvNeXt_V2_Co-designing_and_Scaling_ConvNets_with_Masked_Autoencoders)
- [DACON 구조물 안정성 대회](https://dacon.io/en/competitions/official/236686/overview/description)
- [DACON 대회 코드공유](https://dacon.io/en/competitions/official/236686/codeshare)
- [Physics-Informed Deep Learning for Structural Response](https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2023.08.011)
- [Enhancing ConvNeXt for Small-Size Image Classification](https://www.researchgate.net/publication/397672676_Enhancing_ConvNeXt_for_efficient_small-size_image_classification)

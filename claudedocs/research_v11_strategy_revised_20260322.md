# Research: 규칙 준수 최선 전략 — v8 기반 SOTA 추출 (수정판)

> Date: 2026-03-22
> Confidence: HIGH

---

## 대회 규칙 요약

| 항목 | 허용 | 비고 |
|------|------|------|
| Train 1,000장 학습 | O | |
| Dev 100장 학습 | O | target domain |
| Test 데이터 학습/pseudo-labeling | **X** | Data Leakage = 실격 |
| Test 데이터 통계 사용 (mean, std 등) | **X** | Data Leakage |
| Test-Time Training (파라미터 업데이트) | **X** | |
| TTA (inference only) | O | 파라미터 변경 없음 |
| 외부 데이터 | O | 2025.06.11 이전, 비상업 라이선스 |
| Pretrained model (ImageNet, CLIP 등) | O | 외부 데이터 범주 |
| simulation.mp4에서 feature 추출 | O | train 데이터의 일부 |

**사용 가능 데이터**: train 1,000 + dev 100 + simulation.mp4 1,000개 + 외부 데이터 + pretrained weights

---

## 규칙 안에서 최선의 전략

### 1. DINOv2 Backbone (규칙 준수, 가장 큰 레버)

**왜 DINOv2인가?**
- v10에서 SigLIP ViT가 실패한 이유: supervised pretrain ViT는 소규모 데이터에서 불안정
- DINOv2는 self-supervised pretrain → **소규모 fine-tuning에서 supervised pretrain을 일관되게 이김**
- 문헌: "frozen DINOv2 + linear head만으로 95.8% 정확도" (1,000장 수준 데이터셋)
- **timm에서 바로 사용 가능**: `vit_base_patch14_dinov2.lvd142m`
- DINOv2 pretrained weight는 외부 데이터 → 규칙 허용

**접근법**: DINOv2 backbone **freeze** + learnable head (v10과 달리 backbone fine-tune X)
- ViT의 소규모 데이터 불안정성을 freeze로 해결
- feature가 이미 범용적이라 head만 학습해도 충분
- 이것이 v10과의 핵심 차이

**timm model**: `vit_base_patch14_reg4_dinov2.lvd142m` (86.6M, register 포함)

### 2. Video Physics Feature 고도화 (규칙 준수)

simulation.mp4는 train 데이터의 일부 → 자유롭게 사용 가능.

현재 활용:
- soft_label: scalar 1개 (unstable prob)
- motion_targets: max_diff, mean_diff, severity_bucket, onset_bucket

**추가 추출 가능한 feature들**:
- Optical flow magnitude map (프레임 간 이동 벡터)
- 구조물 contour 변화율 (shape descriptor 시계열)
- 붕괴 시작 프레임 (정확한 onset frame, 현재는 bucket)
- 최대 가속도 시점/크기
- 구조물 높이 변화 곡선
- 마지막 프레임에서 구조물 존재 비율

→ 이들을 auxiliary input으로 모델에 넣거나, 더 정교한 soft target으로 KD에 활용

### 3. 다중 Backbone 앙상블 (규칙 준수)

v8 코드를 복사, backbone만 교체:

| Model | timm name | 특징 | 규칙 |
|-------|-----------|------|------|
| ConvNeXt-V2-Base (v8) | `convnextv2_base.fcmae_ft_in22k_in1k` | 현재 best | O |
| EfficientNetV2-S (v4) | `tf_efficientnetv2_s.in21k_ft_in1k` | 이미 있음 | O |
| **DINOv2-Base** | `vit_base_patch14_reg4_dinov2.lvd142m` | frozen backbone | O |
| **EVA-02-Small** | `eva02_small_patch14_336.mim_in22k_ft_in1k` | MIM pretrain | O |
| **ConvNeXt-CLIP** | `convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384` | CLIP pretrain | O |

5개 모델 앙상블 → Stacking (OOF → LR 2nd level)

### 4. Domain Adaptation — FDA (규칙 준수)

FDA는 dev 이미지의 주파수 통계를 train augmentation에 사용.
- dev는 학습 데이터로 허용됨 → dev 통계 사용도 허용
- test 통계는 사용하지 않음

### 5. Dev-focused Fine-tuning (규칙 준수)

- Stage 1: train+dev 통합 학습 (현재 방식)
- Stage 2: dev 100장에 대해 low-lr fine-tuning (3~5 epoch)
- dev = test와 같은 도메인 → target domain 적응

### 6. Dev-based Global Calibration (규칙 준수)

- per-fold temperature 대신, dev 100장으로 single global temperature fitting
- dev가 test domain이므로 더 정확한 calibration

---

## 실행 순서 (우선순위)

| # | 무엇 | 시간 | 기대 LB 개선 |
|---|------|------|-------------|
| 1 | v4+v8 앙상블 제출 (이미 있음) | 5분 | 0.030~0.033 |
| 2 | DINOv2 frozen backbone 모델 학습 | 2시간 | - |
| 3 | EVA-02 / ConvNeXt-CLIP 모델 학습 | 4시간 | - |
| 4 | 5-model Stacking 앙상블 | 1시간 | 0.018~0.025 |
| 5 | FDA augmentation + v8 재학습 | 2시간 | 0.015~0.022 |
| 6 | Video physics feature 고도화 | 3시간 | 0.013~0.020 |
| 7 | Dev-focused fine-tuning | 1시간 | 0.012~0.018 |
| 8 | Dev-based calibration | 30분 | 0.010~0.016 |

**현실적 최선 목표: LB 0.010~0.016**

---

## Sources

- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2)
- [DINOv2 timm model](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m)
- [DINOv2 Fine-tuning vs Transfer Learning](https://debuggercafe.com/dinov2-for-image-classification-fine-tuning-vs-transfer-learning/)
- [Self-supervised pretraining for small datasets](https://www.nature.com/articles/s41598-023-46433-0)
- [DACON 대회 페이지](https://dacon.io/en/competitions/official/236686/overview/description)
- [DACON Data Leakage 규칙](https://dacon.io/en/competitions/official/236055/talkboard/407731)
- [Pseudo-Labeling 규칙 위반 관련](https://dacon.io/en/forum/405827)

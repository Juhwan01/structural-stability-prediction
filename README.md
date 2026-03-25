# Structural Stability Prediction

> **DACON 236686** | 월간 데이콘 구조물 안정성 물리 추론 AI 경진대회

시뮬레이션 구조물의 2가지 시점(front, top) 이미지로부터 **10초 후 안정성(stable/unstable) 확률**을 예측하는 이미지 분류 모델

---

## Challenge

| 항목 | 내용 |
|------|------|
| **대회** | [구조물 안정성 물리 추론 AI 경진대회](https://dacon.io/competitions/official/236686) |
| **평가 지표** | LogLoss (`unstable_prob`, `stable_prob`) |
| **핵심 난이도** | Train과 Test의 **Domain Shift** — 조명, 카메라 좌표가 완전히 다름 |
| **데이터** | Train 1,000장 (고정 환경) + Dev 100장 (랜덤 환경) + Test 1,000장 (랜덤 환경) |

### Domain Shift 분석

이 대회의 핵심 난이도. Train은 고정된 "실험실 환경", Dev/Test는 랜덤 "실제 환경":

| 특성 | Train | Dev / Test | 차이 |
|------|-------|------------|------|
| 색조 (Hue) | 71.6 | 99.0 | 완전히 다른 톤 |
| 밝기 (Value) | 235.9 ± 3.6 | 211.0 ± 12.1 | 더 어둡고 분산 큼 |
| 조명 위치 | (0.50, 0.29) | (0.56, 0.40) | 이동 |
| RGB 거리 | Train ↔ Dev: 43.6 | Dev ↔ Test: 2.9 | Test ≈ Dev |

---

## Solution

### Architecture: Dual-Stream ConvNeXt-V2

```
Front Image ─→ [ConvNeXt-V2-Base] ─→ GeM ─→ Projection ─┐
                                                          ├─→ Transformer Fusion ─→ Classifier ─→ P(unstable)
Top Image   ─→ [ConvNeXt-V2-Base] ─→ GeM ─→ Projection ─┘
```

- **Backbone**: `convnextv2_base.fcmae_ft_in22k_in1k` (ImageNet-22k pretrained)
- **Pooling**: Generalized Mean (GeM) Pooling
- **Fusion**: 2-layer Transformer Encoder (cross-attention between views)
- **Auxiliary Tasks**: motion regression, onset/severity classification

### Domain Adaptation: FDA (Fourier Domain Adaptation)

Train 이미지의 저주파(색감/조명)를 Dev 이미지의 저주파로 교체하여 도메인 차이 해소:

```
Train Image (밝고 노란 톤) ──FFT──→ Amplitude + Phase
Dev Image   (어둡고 파란 톤) ──FFT──→ Amplitude
                                          │
     Low-frequency amplitude 교체 (beta=0.01)
                                          │
FDA Image (Dev 색감 + Train 구조물) ←──IFFT──┘
```

- `beta=0.01`에서 Hue 75→97 달성 (Target: ~100), 아티팩트 없음
- Train 이미지 100% FDA 적용, Dev는 이미 target domain이라 그대로 사용

### Training Strategy

| 기법 | 설정 |
|------|------|
| Knowledge Distillation | Video teacher soft labels (alpha=0.7) |
| Augmentation | ColorJitter, Perspective, Affine, CLAHE, CoarseDropout |
| Optimizer | AdamW (backbone 1e-5, head 1e-4, wd=1e-4) |
| Scheduler | Cosine Warmup (2 epochs) |
| EMA | decay=0.9995 |
| Validation | Stratified 5-Fold CV (Train 1000 + Dev 100 = 1100) |
| Precision | Mixed Precision (fp16) |
| Early Stopping | patience=7 |
| TTA | HorizontalFlip + Multi-scale (384, 448, 512) |

---

## Experiment Log

| Version | 핵심 변경 | CV LogLoss |
|---------|----------|------------|
| v1 | Baseline ConvNeXt-V2 | 0.1296 |
| v3 | EfficientNetV2-S | 0.3685 |
| v4 | Video KD teacher | - |
| v8 | ConvNeXt-V2 + KD + multi-task | 0.0242 |
| v12 | FDA (beta=0.03, online) | 0.0242 |
| v13a | Strong ColorJitter | 0.0242 |
| v14 | Dev-balanced fold | 0.0242 |
| **v15** | **FDA offline (beta=0.01) + 5-fold** | **진행중** |

> **Note**: CV=0.024는 Train+Dev 혼합 fold 기준. 실제 Dev-only 평가 시 ~0.49 수준으로, LB와의 갭이 존재합니다.

---

## Project Structure

```
structural-stability-prediction/
├── data/
│   ├── train/TRAIN_0001~1000/    # front.png, top.png, simulation.mp4
│   ├── dev/DEV_001~100/          # front.png, top.png (no video)
│   ├── test/TEST_0001~1000/      # front.png, top.png
│   ├── fda_beta001/              # FDA 변환된 train 이미지 (cached)
│   ├── train.csv, dev.csv
│   ├── soft_labels.csv           # Video-based soft labels
│   ├── motion_targets.csv        # Auxiliary task targets
│   └── video_teacher_soft_labels.csv
├── notebooks/
│   ├── eda.ipynb                 # 탐색적 데이터 분석
│   ├── 01_video_soft_labels.ipynb
│   ├── 02~14_*.ipynb             # 실험 v2~v14
│   └── 15_fda_v15.py             # FDA domain adaptation (current)
├── outputs/                      # 모델 체크포인트 (.pt)
├── submissions/                  # 제출 파일들
├── claudedocs/                   # 리서치 노트
├── prompt_plan.md                # 구현 계획
├── run_inference_v14.py          # v14 추론 스크립트
└── pyproject.toml                # uv 프로젝트 설정
```

---

## Quick Start

```bash
# 1. 환경 설정
uv sync

# 2. FDA 전처리 + 5-fold 학습 + 추론
uv run python notebooks/15_fda_v15.py

# 3. 제출 파일 확인
ls submissions/v15b_fda_5fold_submission.csv
```

### Requirements

- Python 3.13+
- CUDA GPU (RTX 3080 10GB 기준 개발)
- ~8GB RAM (num_workers=0 권장)

### Key Dependencies

```
torch, torchvision, timm, albumentations
pandas, numpy, scikit-learn, opencv-python, tqdm
```

---

## Approach Details

### 1. Video-based Soft Label Extraction

Train 데이터의 simulation.mp4에서 물리 시뮬레이션 영상을 분석하여 soft label 추출:
- frame_0 vs frame_last 간 구조물 변위(displacement) 측정
- displacement → soft probability 매핑 (sigmoid)
- Knowledge Distillation의 teacher signal로 활용

### 2. Preprocessing Pipeline

- **Center Physics Crop**: 구조물 중심 영역만 크롭 (front: 25~75% width, top: 29~71%)
- **Checkerboard Rotation Normalization**: Hough Line으로 체커보드 격자 각도 추정 → 정규화
- **FDA Transfer**: Train 이미지의 저주파를 Dev 이미지에서 가져와 색감 통일

### 3. Multi-task Learning

주 태스크(안정성 분류) 외 보조 태스크를 동시 학습하여 feature 품질 향상:
- **Motion Regression**: 구조물 변위량 예측
- **Onset Classification**: 붕괴 시작 시점 분류 (4 buckets)
- **Severity Classification**: 붕괴 심각도 분류 (4 buckets)

---

## Environment

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA RTX 3080 (10GB) |
| OS | Windows 11 |
| Python | 3.13 (uv managed) |
| Framework | PyTorch + timm |

---

## References

- [FDA: Fourier Domain Adaptation for Semantic Segmentation](https://arxiv.org/abs/2004.05498) - Yang & Soatto, CVPR 2020
- [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.01808)
- [DACON 236686 대회 페이지](https://dacon.io/competitions/official/236686)

---

## License

This project is for the DACON competition. Code is provided as-is for educational purposes.

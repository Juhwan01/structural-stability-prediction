# 구조물 안정성 예측 AI - 구현 계획 (All-in-One)

> 상태: **확인 대기중**
> 전략: 처음부터 최종 무기 (점진적 X, 풀 스펙 O)
> 대회: 월간 데이콘 구조물 안정성 물리 추론 AI 경진대회
> 평가: LogLoss (unstable_prob, stable_prob)
> 환경: RTX 3080 10GB, Python, uv, PyTorch

---

## 핵심 철학

> "baseline을 만들고 개선"이 아니라,
> "1개 모델을 완벽하게 → config만 바꿔서 복제 → 앙상블"

---

## Phase 0: 셋업 + 영상 전처리 (선행 작업)

모든 학습에 앞서 영상에서 soft label과 추가 데이터를 추출한다.

```
[0-1] uv 프로젝트 초기화 + 의존성 설치
      torch, torchvision, timm, albumentations, pandas, numpy,
      scikit-learn, opencv-python, matplotlib, tqdm

[0-2] EDA (최소한만)
      - 라벨 분포 확인 (train/dev stable:unstable 비율)
      - 이미지 해상도 확인
      - Train vs Dev 이미지 도메인 차이 시각적 확인

[0-3] 시뮬레이션 영상 → Soft Label 추출 ★핵심★
      1,000개 영상 각각:
      a) frame_0 (초기 상태)과 frame_last (10초 후) 추출
      b) 구조물 영역 마스크 (배경=체커보드 제거)
      c) 두 프레임 간 pixel displacement 측정
         - 구조물 마스크의 IoU 변화
         - 또는 마스크 중심점 이동 거리
      d) displacement → soft_unstable_prob 변환
         - displacement=0 → soft_prob=0.0 (완전 안정)
         - displacement≥threshold → soft_prob=1.0 (완전 붕괴)
         - 중간값 → 선형/시그모이드 매핑
      e) soft_labels.csv 저장: id, soft_unstable_prob

[0-4] 영상 프레임 추가 데이터 추출
      - 각 영상에서 t=0초 프레임 → front view 추가 이미지로 활용
      - data/video_frames/ 에 저장
```

**Phase 0 산출물**:
- `data/soft_labels.csv` (1,000개 soft label)
- `data/video_frames/` (추가 학습 이미지)

---

## Phase 1: 최강 단일 모델 완성 ★핵심 Phase★

처음부터 모든 기법을 적용한 완전체 모델을 만든다.
이 코드가 완성되면, 이후 Phase 2는 config만 바꾸면 된다.

```
[1-1] src/config.py — 모든 설정을 1개 파일에 집중
      @dataclass Config:
        # Model
        backbone: str = "convnextv2_base.fcmae_ft_in22k_in1k_384"
        img_size: int = 384
        num_classes: int = 2
        fusion: str = "late"  # "late" | "channel_concat"

        # Training
        epochs: int = 30
        batch_size: int = 8
        grad_accum: int = 4  # effective bs=32
        lr_backbone: float = 1e-4
        lr_head: float = 1e-3
        weight_decay: float = 0.01
        scheduler: str = "cosine_warmup"
        warmup_epochs: int = 2
        early_stopping_patience: int = 7

        # Regularization
        label_smoothing: float = 0.05
        mixup_alpha: float = 0.3
        cutmix_alpha: float = 1.0
        mixup_prob: float = 0.5  # mixup vs cutmix 선택 확률
        drop_path_rate: float = 0.2

        # Knowledge Distillation
        use_kd: bool = True
        kd_alpha: float = 0.3  # soft label 가중치
        kd_temperature: float = 3.0

        # Domain Augmentation
        aug_brightness: float = 0.4
        aug_contrast: float = 0.4
        aug_saturation: float = 0.4
        aug_hue: float = 0.15
        aug_perspective: float = 0.08
        aug_rotate: int = 20
        aug_shear: int = 15

        # Fold
        n_folds: int = 5
        seed: int = 42

        # Paths
        data_dir: str = "data"
        output_dir: str = "outputs"

[1-2] src/dataset.py — 풀 스펙 Dataset
      - StructuralDataset:
        · front.png + top.png 동시 로딩
        · soft_labels.csv에서 KD soft label 로딩
        · Train/Dev 모두 포함 (is_dev 플래그로 구분)
      - get_train_transforms(): 도메인 적응형 풀 증강
        · 조명: RandomBrightnessContrast, ColorJitter, RandomShadow,
                RandomGamma, RandomToneCurve
        · 카메라: Perspective, Affine(rotate/shear/scale/translate)
        · 기본: HorizontalFlip, GaussianBlur, GaussNoise
        · 정규화: ImageNet mean/std
        · front와 top에 동일 기하 변환 적용 (일관성 유지)
      - get_val_transforms(): Resize + Normalize만
      - Mixup/CutMix는 DataLoader 레벨에서 적용

[1-3] src/model.py — Dual-Stream Late Fusion
      class DualStreamModel(nn.Module):
        - self.backbone_front = timm.create_model(backbone, pretrained=True,
            num_classes=0)  # feature extractor
        - self.backbone_top = timm.create_model(backbone, pretrained=True,
            num_classes=0)
        - Gradient checkpointing 활성화
        - self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
          )
        - forward(front, top) → logits (2-d)

[1-4] src/train.py — 완전체 학습 루프
      - Stratified K-Fold (Train+Dev 통합, 1,100개)
      - Mixed Precision (torch.amp)
      - Gradient Accumulation
      - Mixup/CutMix 랜덤 선택
      - Loss = (1 - kd_alpha) * CE(logits, hard_label, label_smoothing)
             + kd_alpha * KL(log_softmax(logits/T), softmax(soft_label/T)) * T²
      - AdamW with differential LR (backbone vs head)
      - CosineAnnealingWarmRestarts + Linear Warmup
      - Early stopping (LogLoss 기준)
      - 각 fold:
        · best_model_{fold}.pt 저장
        · OOF 예측 저장 (oof_preds_{backbone}_{fold}.npy)
        · Test 예측 저장 (test_preds_{backbone}_{fold}.npy)
      - 전체 CV LogLoss 출력

[1-5] src/inference.py — 추론 + 제출 생성
      - 5-Fold 모델 로딩 → Test 이미지 예측
      - TTA 적용: HorizontalFlip + 밝기 변동 (5회)
      - 5-Fold × 5-TTA = 25개 예측 평균
      - submission.csv 생성 (id, unstable_prob, stable_prob)
```

**Phase 1 완료 기준**: ConvNeXt-V2-Base 5-Fold CV LogLoss 측정, 첫 제출 가능

---

## Phase 2: Backbone 복제 학습 (config만 변경)

Phase 1의 코드를 **그대로** 사용하고, config.backbone만 바꿔서 실행.

```
[2-1] Model B: efficientnetv2_rw_s.ra2_in1k
      python src/train.py --backbone efficientnetv2_rw_s.ra2_in1k --img_size 384

[2-2] Model C: swinv2_base_window12to16_192to256.ms_in22k_ft_in1k_256
      python src/train.py --backbone swinv2_base_... --img_size 256

[2-3] Model D: eva02_small_patch14_336.mim_in22k_ft_in1k
      python src/train.py --backbone eva02_small_... --img_size 336

[2-4] Model E: convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384
      python src/train.py --backbone convnext_base.clip_... --img_size 384

[2-5] (선택) Model F: caformer_b36.sail_in22k_ft_in1k_384
      python src/train.py --backbone caformer_b36... --img_size 384

각 모델에서 저장:
  - outputs/{backbone}/oof_preds.npy (1,100 × 2)
  - outputs/{backbone}/test_preds.npy (1,000 × 2)
  - outputs/{backbone}/cv_score.txt
```

**Phase 2 완료 기준**: 5개+ 모델의 OOF + Test 예측 파일 확보

---

## Phase 3: 앙상블 + 보정 + 최종 제출

```
[3-1] src/ensemble.py — 최적 앙상블
      a) 모든 모델의 OOF 예측 로딩
      b) 모델 간 예측 상관관계 분석 (상관 낮을수록 앙상블 가치↑)
      c) scipy.optimize.minimize로 LogLoss 최소화 가중치 탐색
         - constraint: weights.sum() = 1, weights >= 0
      d) Weighted Average 앙상블 CV LogLoss 출력

[3-2] Temperature Scaling 보정
      - 앙상블 후 OOF logits에서 최적 T 탐색
      - scipy.optimize.minimize_scalar(log_loss_fn, bounds=(0.5, 5.0))
      - 보정 전후 CV LogLoss 비교

[3-3] Test-Time Augmentation (이미 inference.py에 포함)
      - 각 모델의 test 예측이 이미 TTA 적용됨
      - 앙상블 가중치 적용

[3-4] 최종 제출 파일
      - 앙상블 + Temperature Scaling 적용
      - np.clip(pred, 1e-15, 1-1e-15) 안전 처리
      - 행별 합 = 1 검증
      - submissions/final_submission.csv 저장
```

**Phase 3 완료 기준**: 최종 CV LogLoss, 제출 파일 완성

---

## 프로젝트 구조

```
structural-stability-prediction/
├── data/
│   ├── train/              # TRAIN_0001~1000
│   ├── dev/                # DEV_001~100
│   ├── test/               # TEST_0001~1000
│   ├── video_frames/       # 영상에서 추출한 프레임 (Phase 0)
│   ├── train.csv
│   ├── dev.csv
│   ├── soft_labels.csv     # 영상 기반 soft label (Phase 0)
│   └── sample_submission.csv
├── src/
│   ├── config.py           # 모든 설정 (backbone, augmentation, KD 등)
│   ├── dataset.py          # Dataset + 풀 스펙 augmentation
│   ├── model.py            # DualStreamModel
│   ├── train.py            # 학습 (CLI args로 backbone 선택)
│   ├── inference.py        # 추론 + TTA
│   ├── ensemble.py         # 앙상블 + Temperature Scaling
│   └── video_preprocess.py # 영상 → soft label + 프레임 추출
├── outputs/
│   ├── convnextv2_base/    # fold별 모델 + OOF/Test 예측
│   ├── efficientnetv2_s/
│   ├── swinv2_base/
│   ├── eva02_small/
│   └── convnext_clip/
├── submissions/
├── notebooks/
│   └── eda.ipynb
├── claudedocs/
├── prompt_plan.md
└── pyproject.toml
```

---

## 핵심 하이퍼파라미터

| 항목 | 값 | 비고 |
|------|------|------|
| Image Size | 384×384 (backbone별 조정) | |
| Batch Size | 8 (effective 32 via grad accum ×4) | 3080 |
| LR (backbone) | 1e-4 | differential LR |
| LR (head) | 1e-3 | |
| Optimizer | AdamW (wd=0.01) | |
| Scheduler | CosineAnnealing + Warmup 2ep | |
| Epochs | 30 | early stop patience=7 |
| Label Smoothing | 0.05 | |
| Mixup/CutMix | 0.3 / 1.0 (50% 확률) | |
| Drop Path | 0.2 | |
| KD alpha | 0.3 | soft:hard = 3:7 |
| KD Temperature | 3.0 | |
| K-Fold | 5 (Stratified) | Train+Dev 통합 |
| Precision | fp16 (AMP) | |
| Grad Checkpoint | True | |
| TTA | 5회 (HFlip + brightness) | |

---

## 리스크

| 리스크 | 영향 | 완화 |
|--------|------|------|
| 영상 soft label 품질이 낮음 | 중간 | kd_alpha를 0으로 설정하면 hard label만 사용 (폴백) |
| 특정 backbone 3080에서 OOM | 낮음 | batch_size 줄이거나 img_size 조정 |
| 앙상블 과적합 | 낮음 | OOF 기반 가중치 + 상한 제약 |

---

## 이전 계획

이전 점진적 계획 (Phase 0~6)은 `claudedocs/` 리서치 문서에 보관.
이 계획은 "처음부터 풀 스펙" 전략으로 대체됨.

---

**이 계획을 확인해주시면 Phase 0부터 구현을 시작하겠습니다.**

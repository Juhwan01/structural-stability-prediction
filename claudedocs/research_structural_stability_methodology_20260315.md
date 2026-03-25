# 구조물 안정성 예측 AI 대회 - 방법론 리서치 리포트

> 작성일: 2026-03-15
> 대회: 월간 데이콘 '구조물 안정성 물리 추론 AI 경진대회'

---

## Executive Summary

**핵심 과제**: 2-view 이미지(front + top)로 구조물의 stable/unstable 확률을 예측하는 이진 분류 문제. **최대 난관은 도메인 시프트** — train(고정 조명/카메라)과 test(랜덤 조명/카메라) 간 환경 차이가 크다.

**추천 전략**: ConvNeXt-V2/EfficientNetV2 기반 Dual-Stream Late Fusion + 공격적 도메인 증강 + 시뮬레이션 영상 활용 Knowledge Distillation + 다중 모델 앙상블

**신뢰도**: 높음 (ShapeStacks 등 선행연구에서 검증된 접근법 + 데이콘 이미지 대회 우승 솔루션 패턴 일치)

---

## 1. 데이터 분석 요약

### 1.1 데이터 구성

| 구분 | 샘플 수 | 이미지 | 영상 | 환경 |
|------|---------|--------|------|------|
| Train | 1,000 | front.png + top.png | simulation.mp4 (10초) | 고정 조명/카메라 |
| Dev | 100 | front.png + top.png | 없음 | 랜덤 조명/카메라 |
| Test | 1,000 | front.png + top.png | 없음 | 랜덤 조명/카메라 |

### 1.2 이미지 특성 (시각 확인 결과)

- **해상도**: ~390x390px
- **배경**: 체커보드 패턴 (흰색/연파란색)
- **구조물**: 다양한 색상의 블록들이 쌓인 형태
- **Front view**: 3D 원근 시점에서 전면 촬영
- **Top view**: 조감도 (새눈 시점)
- **Unstable 예시 (TRAIN_0001)**: 넓은 바닥에서 점점 좁아지는 피라미드형, 꼭대기 블록들이 불안정하게 편중
- **Stable 예시 (TRAIN_0012)**: 균일한 직육면체형, 무게중심이 중앙에 위치
- **핵심 시각적 단서**: 높이/폭 비율, 무게중심 편향, 돌출 블록, 블록 정렬도

### 1.3 도메인 시프트 (핵심 문제)

- **Train**: 조명과 카메라 고정 → 그림자 패턴, 시점 일정
- **Dev/Test**: 조명과 카메라 랜덤 → 그림자 방향/강도 변동, 시점 변동
- **의미**: 조명/시점에 의존하는 features를 학습하면 test에서 성능 급락

### 1.4 라벨 분포

- Train: 약 50:50으로 추정 (균형 데이터셋) — 정확한 비율은 EDA에서 확인 필요
- Dev: 100개 중 unstable 비율 확인 필요
- 평가: LogLoss (확률 예측 품질이 핵심)

---

## 2. 핵심 도전 과제

1. **도메인 시프트**: train→test 환경 차이가 가장 큰 장벽
2. **소량 데이터**: train 1,000 + dev 100 = 총 1,100개 (딥러닝에는 적은 양)
3. **물리 추론**: 단순 외형 분류가 아닌, 구조적 안정성의 물리적 원리 이해 필요
4. **경계 샘플**: 외형만으로 구분 어려운 boundary 케이스 존재
5. **확률 보정**: LogLoss 평가이므로 잘 보정된(calibrated) 확률 출력 필수

---

## 3. 추천 방법론 (Tier별)

### Tier 1: 메인 전략 (최우선 구현)

#### 3.1 Dual-Stream Late Fusion Architecture

```
Front Image → Backbone A → Feature Vector (768-d)
                                              ↘
                                               Concat → FC Layers → P(unstable), P(stable)
                                              ↗
Top Image   → Backbone B → Feature Vector (768-d)
```

**왜 Late Fusion인가?**
- Front와 Top은 완전히 다른 시점 → 독립적 feature 추출이 효과적
- Early fusion(채널 concat)은 소량 데이터에서 오히려 혼란 초래
- 선행연구(Multi-view Classification, WACV 2024)에서 Late Fusion이 소량 데이터에서 우수

**Backbone 선택지 (우선순위순)**:

| 모델 | 장점 | 단점 | 추천도 |
|------|------|------|--------|
| **ConvNeXt-V2-Small** | ImageNet-22K pretrain, 강건한 feature, modern CNN | 약간 무거움 | ★★★★★ |
| **EfficientNetV2-S** | 효율적, 검증된 성능 | V1 대비 개선폭 제한적 | ★★★★☆ |
| **EVA-02-Small (ViT)** | 강력한 표현력, MAE pretrain | 소량 데이터에서 과적합 위험 | ★★★☆☆ |
| **Swin-V2-Small** | 계층적 feature, 해상도 유연 | 학습 불안정 가능 | ★★★☆☆ |

**추천**: ConvNeXt-V2-Small을 메인으로, EfficientNetV2-S를 앙상블 후보로

#### 3.2 도메인 적응형 증강 (Domain-Adaptive Augmentation)

**이것이 승패를 가른다.** Train(고정환경) → Test(랜덤환경) 갭을 줄이는 것이 핵심.

```python
# 핵심 증강 전략
augmentation = A.Compose([
    # 1. 조명 변동 시뮬레이션 (가장 중요)
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    A.RandomGamma(gamma_limit=(70, 130), p=0.5),
    A.RandomShadow(p=0.3),  # 그림자 방향 변동

    # 2. 카메라 시점 변동 시뮬레이션 (매우 중요)
    A.Affine(
        scale=(0.85, 1.15),
        translate_percent=(-0.1, 0.1),
        rotate=(-15, 15),
        shear=(-10, 10),
        p=0.7
    ),
    A.Perspective(scale=(0.02, 0.08), p=0.5),

    # 3. 일반 증강
    A.HorizontalFlip(p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(var_limit=(5, 25), p=0.2),

    # 4. 정규화
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**핵심 포인트**:
- 조명/카메라 관련 증강을 **매우 공격적으로** 적용
- 블록 구조 자체의 기하학적 관계는 보존해야 하므로, 과도한 기하 변환은 주의
- Dev 세트 이미지를 참고하여 증강 범위 설정

#### 3.3 시뮬레이션 영상 활용 (Knowledge Distillation)

Train에만 제공되는 10초 시뮬레이션 영상을 활용하는 **2단계 전략**:

**Stage 1: Video Teacher Model**
```
simulation.mp4 → Frame Sampling (N frames) → Video/3D-CNN or Frame-level CNN
→ Teacher가 stable/unstable을 높은 정확도로 학습
```
- 영상에서 **실제 붕괴 여부**를 직접 관찰 가능 → 라벨 정확도 검증 + 소프트 라벨 생성
- 영상의 초기 프레임(0~1초)은 이미지와 유사 → 추가 학습 데이터로 활용 가능

**Stage 2: Knowledge Distillation**
```
Teacher(video model) → Soft Labels (확률 분포)
Student(image-only model) → Teacher의 소프트 라벨로 학습
```
- Teacher의 소프트 라벨이 hard label보다 더 풍부한 정보 제공
- 경계 샘플에서 특히 효과적 (0.6 vs 0.4 같은 미묘한 확률)

**대안: 영상 프레임을 학습 데이터로 활용**
- 각 영상에서 초기 1~2초 구간의 프레임을 추가 이미지로 사용
- 1,000 영상 × 5프레임 = 5,000개 추가 학습 이미지
- 단, 프레임 간 유사도가 높아 중복 효과에 주의

### Tier 2: 성능 부스팅 전략

#### 3.4 앙상블 (Ensemble)

LogLoss 평가에서 앙상블은 거의 필수:

```
Model 1: ConvNeXt-V2 + Dual-Stream
Model 2: EfficientNetV2 + Dual-Stream
Model 3: Swin-V2 + Dual-Stream
Model 4: ConvNeXt-V2 + Channel Concat (Early Fusion)
→ Weighted Average or Stacking
```

- 최소 3~5개 모델 앙상블 권장
- 각 모델은 서로 다른 backbone 또는 다른 fusion 전략 사용
- **Stacking**(2nd-level model)보다 **Weighted Average**가 소량 데이터에서 더 안전

#### 3.5 Test-Time Augmentation (TTA)

```python
# 추론 시 N번 증강 후 평균
predictions = []
for _ in range(10):
    aug_images = apply_tta_augmentation(images)
    pred = model(aug_images)
    predictions.append(pred)
final_pred = np.mean(predictions, axis=0)
```

- HorizontalFlip, 밝기 변동, 약한 기하 변환 등
- 5~10회 TTA로 LogLoss 개선 기대

#### 3.6 확률 보정 (Probability Calibration)

LogLoss는 확률 보정 품질에 민감:

```python
from sklearn.calibration import CalibratedClassifierCV
# Temperature Scaling이 가장 효과적
# Dev set을 calibration set으로 활용
```

- **Temperature Scaling**: 단일 파라미터로 모델 출력 보정
- Dev 100개를 calibration에 활용 (또는 K-Fold 내 validation으로)

### Tier 3: 추가 실험 후보

#### 3.7 Hand-crafted Physical Features

이미지에서 물리적 특성을 직접 추출하여 보조 feature로 활용:

- **높이/폭 비율**: 구조물의 종횡비 (높을수록 불안정)
- **무게중심 편향**: Top view에서 블록 분포의 비대칭도
- **Bounding box 면적 비율**: Front vs Top 면적 비교
- **블록 수 추정**: 색상 분포 기반 블록 개수
- 이를 CNN feature에 concat하여 최종 분류기 입력

#### 3.8 Contrastive Learning / Self-Supervised Pre-training

- Dev+Test의 unlabeled 이미지(1,100장)로 SimCLR/BYOL 사전학습
- 도메인 적응에 효과적일 수 있으나, 데이터 양이 적어 효과 불확실

#### 3.9 외부 데이터 활용

대회 규칙상 외부 데이터 사용 가능:
- **ShapeStacks** 데이터셋: 20,000개의 블록 구조물 안정성 데이터 (가장 유사)
- **PhyDNet / IntPhys**: 물리 추론 벤치마크 데이터
- 시뮬레이터 자체 제작: Blender + Bullet Physics로 유사 데이터 생성

---

## 4. 학습 전략

### 4.1 K-Fold Cross Validation

```
Train(1,000) + Dev(100) = 1,100 samples
→ Stratified 5-Fold CV
→ 각 Fold의 validation에 Dev 샘플이 포함되도록 구성
```

**중요**: Dev 데이터를 반드시 학습에 포함. Dev는 test와 동일한 환경이므로 **도메인 적응 학습**에 필수적.

### 4.2 학습 하이퍼파라미터 (권장)

| 항목 | 값 | 근거 |
|------|------|------|
| Optimizer | AdamW | 소량 데이터에서 안정적 |
| LR | 1e-4 (backbone), 1e-3 (head) | pretrained backbone은 낮은 LR |
| Scheduler | CosineAnnealingWarmRestarts | 소량 데이터에서 과적합 방지 |
| Epochs | 30~50 | Early stopping과 함께 |
| Batch Size | 16~32 | 메모리 여유 시 32 |
| Weight Decay | 0.01~0.05 | 과적합 방지 |
| Label Smoothing | 0.05~0.1 | LogLoss에서 극단 확률 방지 |
| Mixup/CutMix | alpha=0.2~0.4 | 소량 데이터 정규화 |
| Image Size | 384x384 | backbone 기본 크기 활용 |

### 4.3 Loss Function

```python
# 기본: CrossEntropyLoss with Label Smoothing
loss = nn.CrossEntropyLoss(label_smoothing=0.1)

# 대안: Focal Loss (경계 샘플에 더 집중)
# 대안: BCE with LogLoss-aligned loss
```

---

## 5. 구현 우선순위 로드맵

### Phase 1: Baseline (1~2일)
1. EDA: 라벨 분포, 이미지 크기 확인
2. 단일 모델 baseline: EfficientNetV2-S + 6채널 concat (front+top)
3. 기본 증강 + 5-Fold CV
4. Dev set LogLoss 확인

### Phase 2: Core Model (2~3일)
1. Dual-Stream Late Fusion 구현
2. ConvNeXt-V2-Small backbone 적용
3. 도메인 적응형 공격적 증강
4. Dev set에서 성능 검증

### Phase 3: Video + Distillation (2~3일)
1. 시뮬레이션 영상에서 프레임 추출
2. Video Teacher 모델 학습
3. Knowledge Distillation 적용
4. 소프트 라벨 기반 Student 학습

### Phase 4: Ensemble + Polish (1~2일)
1. 다중 모델 학습 (3~5개)
2. Weighted Average 앙상블
3. Temperature Scaling 보정
4. TTA 적용
5. 최종 제출 파일 생성

---

## 6. 기술 스택 권장

```
Python 3.10+
uv (패키지 관리)
PyTorch 2.x
timm (pretrained models)
albumentations (증강)
scikit-learn (평가/보정)
pandas, numpy
opencv-python (이미지/비디오 처리)
wandb or tensorboard (실험 추적)
```

---

## 7. 리스크 분석

| 리스크 | 영향 | 완화 방안 |
|--------|------|-----------|
| 도메인 시프트로 test 성능 급락 | 높음 | 공격적 증강 + Dev 포함 학습 |
| 소량 데이터 과적합 | 높음 | pretrained model + 강한 정규화 |
| 경계 샘플 분류 실패 | 중간 | KD soft labels + 영상 프레임 활용 |
| 확률 보정 불량 (LogLoss 악화) | 중간 | Temperature Scaling + Label Smoothing |
| 앙상블 과적합 | 낮음 | K-Fold로 다양성 확보 |

---

## Sources

- [ShapeStacks: Learning Vision-Based Physical Intuition (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Oliver_Groth_ShapeStacks_Learning_Vision-Based_ECCV_2018_paper.pdf)
- [Visual Stability Prediction and Its Application to Manipulation](https://ar5iv.labs.arxiv.org/html/1609.04861)
- [Multi-View Classification Using Hybrid Fusion and Mutual Distillation (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/papers/Black_Multi-View_Classification_Using_Hybrid_Fusion_and_Mutual_Distillation_WACV_2024_paper.pdf)
- [Multi-Scale Feature Fusion for Dual-View X-ray Inspections](https://arxiv.org/html/2502.01710v2)
- [ConvNeXt Architecture Review (2025)](https://www.researchgate.net/publication/396787830)
- [DACON 구조물 안정성 물리 추론 AI 경진대회](https://dacon.io/en/competitions/official/236686/overview/description)
- [Image Classification State-of-the-Art 2025](https://opencv.org/blog/image-classification/)

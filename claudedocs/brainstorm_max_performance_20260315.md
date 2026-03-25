# 구조물 안정성 예측 - 최고 성능 브레인스토밍

> 목표: 대회 최상위 성적 달성
> 작성일: 2026-03-15

---

## 이 대회의 본질: 3가지 핵심 축

1. **도메인 시프트 극복** — Train(고정환경) → Test(랜덤환경) 갭을 줄이는 자가 최강
2. **영상 데이터 최대 활용** — Train에만 있는 시뮬레이션 영상은 독보적 우위 수단
3. **확률 보정 (LogLoss)** — 맞추는 것보다 "얼마나 확신하는가"가 점수를 결정

---

## 전략 A: 모델 아키텍처 (다다익선 앙상블)

### A1. Backbone Pool (최소 5종 이상)

| # | 모델 | Pretrain | 파라미터 | 특징 |
|---|------|----------|---------|------|
| 1 | ConvNeXt-V2-Large | ImageNet-22K | 198M | 현대 CNN 최강, 로컬 피처 |
| 2 | EVA-02-Large | CLIP+IN-22K | 304M | ViT 계열 최고 성능 90.0% IN |
| 3 | SwinV2-Base | ImageNet-22K | 88M | 계층적 ViT, 해상도 유연 |
| 4 | EfficientNetV2-L | ImageNet-21K | 118M | 효율적이면서 강력 |
| 5 | BEiT-3-Base | 멀티모달 pretrain | 86M | 멀티모달 표현력 |
| 6 | InternImage-Base | ImageNet-22K | 97M | Deformable Conv, 기하학적 특징에 강점 |

**왜 대형 모델인가?**
- 1,100개는 적지만, ImageNet-22K pretrained 대형 모델은 소량 fine-tuning에서 오히려 소형보다 우수
- 대형 모델이 더 범용적인 feature를 가지고 있어 도메인 시프트에 강건

### A2. Multi-View Fusion 전략 (3가지 모두 실험)

```
Strategy 1: Late Fusion (독립 추출 → concat)
  Front → Backbone → [768-d] ─┐
                               ├─ Concat → MLP Head
  Top   → Backbone → [768-d] ─┘

Strategy 2: Cross-Attention Fusion (시점 간 상호작용)
  Front → Backbone → Tokens ─┐
                              ├─ Cross-Attention → CLS → Head
  Top   → Backbone → Tokens ─┘

Strategy 3: Channel Concat (6채널 입력)
  [Front(3ch) | Top(3ch)] → 6ch Conv1 → Backbone → Head
```

- Late Fusion이 가장 안전하지만, **Cross-Attention이 성능 상한이 높음**
- 물리적 안정성은 front+top의 관계(높이 vs 바닥면적)가 핵심 → Cross-Attention이 이 관계를 학습 가능

---

## 전략 B: 시뮬레이션 영상 극한 활용 (독보적 우위)

### B1. Video Teacher → Image Student (Knowledge Distillation)

```
Phase 1: Video Teacher 학습
  simulation.mp4 → VideoMAE / TimeSformer / SlowFast
  → 영상 전체를 보고 stable/unstable 판별 (거의 100% 정확도 가능)
  → Soft probability 출력: [0.73, 0.27] 같은 미묘한 확률

Phase 2: Soft Label로 Image Student 학습
  Image model이 Teacher의 소프트 라벨로 학습
  → hard label [1,0] 대신 [0.73, 0.27]로 학습
  → 경계 샘플에서 특히 큰 이득
```

**핵심**: Teacher가 "이 구조물은 73% 확률로 불안정"이라고 판단하면, Student도 이 미묘함을 학습

### B2. 영상 프레임 → 대규모 학습 데이터 생성

```
각 영상(10초, ~30fps) → 300프레임
1,000 영상 × 300프레임 = 300,000 이미지 (잠재적)

실용적 전략:
- 초기 1초 (프레임 0~30): 정적 상태 → front/top 이미지와 유사, 학습 보조 데이터
- 중기 3~5초: 붕괴 시작 단계 → 불안정 신호가 가장 풍부
- 시점이 front view이므로 top view와는 다르지만, 단일 시점 모델 학습에 활용 가능

프레임에서 추출 가능한 추가 정보:
- optical flow → 움직임 방향/크기
- 프레임 간 차이(frame diff) → 어떤 부분이 먼저 움직이는지
```

### B3. 영상 기반 Label Refinement

```
simulation.mp4의 최종 프레임을 직접 분석:
- 마지막 프레임에서 구조물이 완전히 붕괴? → 확실한 unstable
- 마지막 프레임에서 미세한 이동만? → 경계 케이스
→ hard label보다 더 정밀한 연속적 라벨 생성 가능

방법:
  frame_0 vs frame_last의 구조물 위치 차이(pixel displacement)를 측정
  → displacement 크기를 연속 값으로 → soft label로 변환
```

---

## 전략 C: 도메인 적응 (Train→Test 갭 극복)

### C1. 공격적 도메인 증강 (Aggressive Domain Augmentation)

Dev 이미지를 분석한 결과, 주요 차이:
- 조명 방향/강도 변동 (그림자 위치)
- 카메라 각도 변동 (시점)
- 전체적인 색온도 변화

```python
# "Train을 Dev처럼 만드는" 증강
domain_aug = A.Compose([
    # 조명 (가장 중요)
    A.RandomBrightnessContrast(0.4, 0.4, p=0.8),
    A.ColorJitter(0.4, 0.4, 0.4, 0.15, p=0.7),
    A.RandomShadow(shadow_roi=(0,0,1,1), p=0.5),
    A.RandomToneCurve(scale=0.3, p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),

    # 카메라 시점
    A.Perspective(scale=(0.02, 0.10), p=0.6),
    A.Affine(scale=(0.8, 1.2), translate_percent=(-0.15, 0.15),
             rotate=(-20, 20), shear=(-15, 15), p=0.7),

    # 체커보드 배경의 패턴 변동 대응
    A.RandomCrop(height=350, width=350, p=0.3),
    A.Resize(384, 384),
])
```

### C2. Style Transfer 기반 도메인 정렬

```
Train 이미지의 스타일 → Dev 이미지의 스타일로 변환
방법: AdaIN (Adaptive Instance Normalization)
  content: Train 이미지 (구조물 형태)
  style: Dev 이미지 (조명/색감)
→ Train 구조물을 Dev 환경에서 찍은 것처럼 변환

장점: 구조적 정보는 보존하면서 도메인만 맞춤
실용성: 1,000 × 100 = 100,000개의 도메인 변환 이미지 생성 가능
```

### C3. Pseudo-Labeling + Self-Training

```
Round 1: Train+Dev로 모델 학습
Round 2: Test 1,000개에 대해 예측
Round 3: 높은 confidence 샘플(>0.95)만 pseudo-label로 추가
Round 4: Train+Dev+Pseudo로 재학습
Round 5: 반복 (2~3회)

주의:
- Confidence threshold를 높게 (0.95+)
- 클래스 비율이 크게 치우치지 않도록 균형 맞춤
- 과적합 모니터링 필수
```

### C4. Test-Time Adaptation (TTA+)

```
단순 TTA: 여러 증강 버전의 평균
고급 TTA: Test 데이터 배치에서 BN 통계 업데이트 (Tent)

Test-Time Self-Training:
  1. 전체 test 데이터의 BN 통계 수집
  2. 높은 confidence 예측으로 pseudo-label 생성
  3. 1-2 epoch 미세 조정
  4. 최종 예측
```

---

## 전략 D: 물리 기반 Feature Engineering

### D1. 이미지에서 물리적 특성 추출

```python
def extract_physics_features(front_img, top_img):
    # 구조물 마스크 추출 (배경 제거)
    front_mask = segment_structure(front_img)  # 체커보드 배경 vs 구조물
    top_mask = segment_structure(top_img)

    # 1. 종횡비 (Height/Width ratio) - 높을수록 불안정
    bbox = get_bounding_box(front_mask)
    aspect_ratio = bbox.height / bbox.width

    # 2. 무게중심 편향 (Center of Mass offset)
    com = center_of_mass(front_mask)
    com_offset = abs(com.x - bbox.center_x) / bbox.width

    # 3. 바닥 지지 면적 비율
    bottom_width = measure_bottom_width(front_mask)
    support_ratio = bottom_width / bbox.width

    # 4. Top view 대칭도
    symmetry = calculate_symmetry(top_mask)

    # 5. 블록 밀도 (top view 면적 / front view 면적)
    density_ratio = top_mask.sum() / front_mask.sum()

    # 6. 상단 돌출도 (위쪽 블록이 바닥보다 넓은 정도)
    overhang = measure_overhang(front_mask)

    return [aspect_ratio, com_offset, support_ratio,
            symmetry, density_ratio, overhang]
```

### D2. 물리 Feature + CNN Feature 결합

```
CNN: [768-d feature from front] + [768-d feature from top] = [1536-d]
Physics: [6-d handcrafted features]
Combined: [1536 + 6] → MLP → prediction

또는 Auxiliary Loss:
  Main: BCE(pred, label)
  Aux: MSE(predicted_physics, extracted_physics)
  → CNN이 물리적 특성도 함께 학습하도록 유도
```

---

## 전략 E: 앙상블 극대화

### E1. 다양성이 핵심

```
앙상블 다양성 확보 축:
1. Backbone 다양성: ConvNeXt vs ViT vs Swin vs EfficientNet
2. Fusion 다양성: Late vs Cross-Attention vs Channel Concat
3. 데이터 다양성: Original only vs +Video frames vs +Style Transfer
4. Loss 다양성: CE vs Focal vs Label Smoothing
5. Resolution 다양성: 224 vs 384 vs 448
6. Fold 다양성: 5-Fold × N models

최종 앙상블 = 20~50개 모델의 weighted average
```

### E2. Stacking (2-Level Ensemble)

```
Level 1: 각 모델의 OOF(Out-of-Fold) 예측
Level 2: LightGBM 또는 Ridge로 최적 가중치 학습

Level 1 features:
  - Model 1 unstable_prob (fold 1~5 OOF)
  - Model 2 unstable_prob (fold 1~5 OOF)
  - ...
  - Physics features (6-d)

Level 2: LogisticRegression 또는 LightGBM
```

### E3. 확률 보정 (최종 단계)

```python
# Temperature Scaling (가장 중요)
# Dev set 또는 OOF predictions에서 최적 T 탐색

import scipy.optimize as opt

def temperature_scale(logits, T):
    return softmax(logits / T)

def find_optimal_T(logits, labels):
    def loss_fn(T):
        scaled = temperature_scale(logits, T)
        return log_loss(labels, scaled)
    result = opt.minimize_scalar(loss_fn, bounds=(0.1, 10), method='bounded')
    return result.x

# Isotonic Regression (대안)
# Platt Scaling (대안)
```

---

## 전략 F: 외부 데이터 & 합성 데이터

### F1. ShapeStacks 데이터셋 활용

- 20,000개의 블록 구조물 안정성 데이터 (CC BY-SA 라이선스)
- 대회 데이터와 가장 유사한 외부 데이터
- Pre-training → Fine-tuning 전략으로 활용

### F2. 시뮬레이터로 합성 데이터 생성

```
Blender + Bullet Physics:
1. 다양한 블록 구조물 생성 (랜덤 배치)
2. 물리 시뮬레이션 실행 → stable/unstable 자동 라벨링
3. 다양한 조명/카메라 각도로 렌더링
→ 수만 개의 추가 학습 데이터 생성 가능

비용: 시뮬레이터 구축에 시간 투자 필요
효과: 도메인 시프트 문제를 근본적으로 해결 가능
```

---

## 성능 기여도 예상 (LogLoss 개선)

| 전략 | 예상 기여도 | 구현 난이도 | 우선순위 |
|------|------------|-----------|---------|
| 대형 pretrained backbone | ★★★★☆ | 낮음 | 1 |
| 도메인 적응형 증강 | ★★★★★ | 낮음 | 1 |
| Dual-Stream Fusion | ★★★☆☆ | 중간 | 2 |
| 영상 KD (soft label) | ★★★★☆ | 중간 | 2 |
| 다모델 앙상블 (5+) | ★★★★★ | 중간 | 2 |
| 확률 보정 (Temp Scaling) | ★★★★☆ | 낮음 | 2 |
| 영상 프레임 추가 데이터 | ★★★☆☆ | 낮음 | 3 |
| Pseudo-labeling | ★★★☆☆ | 중간 | 3 |
| Style Transfer 도메인 정렬 | ★★☆☆☆ | 높음 | 4 |
| 물리 Feature Engineering | ★★☆☆☆ | 높음 | 4 |
| Cross-Attention Fusion | ★★★☆☆ | 높음 | 4 |
| 외부 데이터 (ShapeStacks) | ★★☆☆☆ | 중간 | 5 |
| 합성 데이터 생성 | ★★☆☆☆ | 매우 높음 | 5 |
| Test-Time Adaptation | ★★☆☆☆ | 높음 | 5 |

---

## 최종 추천: 단계별 실행 계획

### Phase 1: Strong Baseline (2일)
- [ ] EDA 완료 (라벨 분포, 이미지 크기, dev vs train 시각 비교)
- [ ] ConvNeXt-V2-Base + Late Fusion + 기본 증강
- [ ] 5-Fold CV (Train+Dev)
- [ ] Dev LogLoss 측정 → baseline 확보

### Phase 2: 핵심 부스팅 (3일)
- [ ] 공격적 도메인 증강 적용
- [ ] 영상 프레임 추출 + 추가 학습 데이터
- [ ] Video Teacher → KD soft label 생성
- [ ] EfficientNetV2-L, EVA-02 추가 학습

### Phase 3: 앙상블 (2일)
- [ ] 5+ 모델 OOF 수집
- [ ] Weighted Average 탐색
- [ ] Stacking 실험
- [ ] Temperature Scaling 보정

### Phase 4: 추가 실험 (여유 시)
- [ ] Pseudo-labeling
- [ ] Cross-Attention Fusion
- [ ] Physics Feature Engineering
- [ ] ShapeStacks 외부 데이터

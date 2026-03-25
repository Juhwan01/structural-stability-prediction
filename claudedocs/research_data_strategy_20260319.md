# Research: 0.058 벽 돌파를 위한 데이터 전략

## Executive Summary

현재 최고 성능: v4 (effv2s + KD + EMA) LB 0.058
핵심 병목: **train/test 도메인 시프트** (이미지 품질 4배 차이)
해결 열쇠: 데이터 자체를 늘리거나 변환하여 test 분포에 가깝게 만드는 것

---

## 1. 현재 보유 데이터 현황

| 자산 | 수량 | 특성 |
|------|------|------|
| Train 이미지 | 1000 (front+top) | 60KB, 저품질 압축 |
| Train 비디오 | 1000 (simulation.mp4) | ~750KB, train만 존재 |
| Dev 이미지 | 100 (front+top) | 245KB, 고품질 |
| Test 이미지 | 1000 (front+top) | 240KB, 고품질 |
| Video Teacher soft labels | 1000 | OOF logits (T=3.0) |
| Motion targets | 1000 | severity/onset buckets |
| Soft labels | 1000 | mean_diff, changed_ratio |

### 핵심 도메인 시프트

| 속성 | Train | Dev/Test |
|------|-------|----------|
| 파일 크기 | ~60KB | ~245KB |
| 압축률 | 7.2x | 1.8x |
| Entropy | 2.96 bits | 4.2 bits |
| 텍스처 | 단순 블록 | 세밀한 텍스처/그라데이션 |
| 렌더링 | 저품질 | 고품질 |

---

## 2. 데이터 확장 전략 (실행 가능성 순)

### Strategy A: Style Transfer Augmentation (높은 가능성)
**개념**: dev/test 스타일을 train에 입히기

- Train 이미지를 dev/test와 비슷한 "고품질 스타일"로 변환
- 방법: AdaIN (Adaptive Instance Normalization) style transfer
  - dev 100장의 스타일 통계(mean, std)를 train 이미지에 적용
  - 추가 모델 학습 불필요, feature-level 변환만으로 가능
- **장점**: 구현 간단, 레이블 유지, domain gap 직접 공략
- **리스크**: 구조 정보가 스타일과 함께 왜곡될 수 있음

### Strategy B: Pseudo-Labeling (Self-Training) (중간 가능성)
**개념**: test 1000장에 pseudo label 부여 → 학습 데이터로 활용

- 현재 v4 모델로 test 예측 → 고신뢰 샘플(prob > 0.95 or < 0.05)만 pseudo label
- test 이미지가 dev/test 도메인이므로, **target domain 데이터 직접 학습 가능**
- iterative self-training으로 점진적 확장
- **장점**: test 도메인 데이터를 직접 학습, 1000장 추가 가능
- **리스크**: 잘못된 pseudo label이 전파될 수 있음 (confirmation bias)
- **완화**: 높은 threshold (>0.95), curriculum learning

### Strategy C: Video Frame Augmentation (중간 가능성)
**개념**: train의 simulation.mp4에서 다양한 프레임 추출 → 이미지 학습 데이터화

- 현재: video는 teacher model 학습에만 사용
- 개선: video의 중간 프레임들을 "시뮬레이션 진행 상태" 이미지로 활용
- 시작~끝 프레임 차이 = 불안정도 시각적 근거
- **장점**: 추가 데이터 없이 train 1000 → 수천~만 장 확장
- **리스크**: 비디오는 train에만 있어 domain shift 해결에는 직접 기여 안 함

### Strategy D: Dev 이미지 활용 극대화 (높은 가능성)
**개념**: dev 100장을 최대한 활용

- 현재: dev를 train과 함께 CV에 포함
- 개선: dev만 따로 fine-tuning 단계 추가 (2-stage training)
  1. Stage 1: train 1000장으로 pretraining
  2. Stage 2: dev 100장으로 few-shot fine-tuning (낮은 lr, 적은 epoch)
- dev가 test와 같은 도메인이므로, dev에서 학습한 feature가 test에 직접 전이
- **장점**: 도메인 정렬 직접 수행
- **리스크**: dev 100장으로 overfitting 가능 → 강한 regularization 필요

### Strategy E: Test-Time Training (TTT) (실험적)
**개념**: test 이미지별로 모델을 약간 adaptation

- Self-supervised loss (e.g., rotation prediction, contrastive)를 test 이미지에 적용
- 각 test 이미지로 몇 step fine-tune 후 예측
- **장점**: 개별 이미지에 맞춤 적응
- **리스크**: 구현 복잡, inference 시간 증가

---

## 3. 우선순위 추천

| 순위 | 전략 | 예상 효과 | 구현 난이도 | 근거 |
|------|------|----------|------------|------|
| 1 | **B: Pseudo-Labeling** | 높음 | 낮음 | test 도메인 데이터 직접 학습 |
| 2 | **D: Dev Fine-tuning** | 중-높 | 낮음 | target 도메인 few-shot 적응 |
| 3 | **A: Style Transfer** | 중간 | 중간 | train→test 스타일 정렬 |
| 4 | **C: Video Frames** | 낮음 | 중간 | 도메인 시프트 해결 안 됨 |
| 5 | **E: TTT** | 불확실 | 높음 | 연구 수준, 실험적 |

### 권장 조합: B + D

1. v4 모델로 test pseudo labels 생성 (고신뢰만)
2. train + pseudo-labeled test로 재학습
3. dev 100장으로 마지막 fine-tuning
4. 이렇게 하면 **train domain + test domain 모두 학습**

---

## 4. v5 실패에서 배운 교훈 (적용 필수)

1. **Per-fold temperature는 유지** - v4에서 효과 입증, 제거하면 LB 악화
2. **EMA 유지** - v4 파이프라인 그대로 사용
3. **Multi-backbone 앙상블은 비효율** - 불안정 backbone이 전체 오염
4. **Dev-only calibration은 역효과** - dev/test 분포 불일치
5. **effv2s가 가장 안정적** - swinv2s는 CV만 좋고 LB에서 열세

---

## Sources

- [A Survey of Data Augmentation in Domain Generalization (2025)](https://link.springer.com/article/10.1007/s11063-025-11747-9)
- [Domain Adaptation Using Pseudo Labels (2024)](https://arxiv.org/html/2402.06809v2)
- [Style Transfer as Data Augmentation (2025)](https://arxiv.org/html/2502.02475)
- [Pseudo Labels for Unsupervised Domain Adaptation: A Review](https://www.mdpi.com/2079-9292/12/15/3325)
- [Semi-Supervised Domain Adaptation via Selective Pseudo Labeling](https://arxiv.org/abs/2104.00319)
- [Enhancing Image Classification in Small and Unbalanced Datasets through Synthetic Data Augmentation](https://arxiv.org/html/2409.10286v2)

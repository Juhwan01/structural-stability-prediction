# Research: v8 "확신하고 틀리는" 문제 해결

> Date: 2026-03-23

## 진짜 문제

```
Train (1000장): 에러율 0.1%, 평균 loss 0.008
Dev (100장):    에러율 5.0%, 평균 loss 0.187  ← 23배 나쁨!

전체 loss의 70%가 dev 100장에서 발생
특히 3건의 "stable을 99% 확신으로 unstable로 오판"이 LogLoss 폭발시킴

원인: train(밝기 218) vs dev(밝기 192) → 모델이 밝은 환경에 과적합
```

## 해결 방향 3가지

### 1. Mixup Augmentation (가장 유력)

**원리**: 두 이미지를 섞어서 학습 → 모델이 극단적 확신을 못 하게 됨

```python
# 이미지 A(unstable) + 이미지 B(stable)를 7:3으로 섞음
mixed_img = 0.7 * img_A + 0.3 * img_B
mixed_label = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
→ 모델이 "100% unstable" 대신 "70% unstable"을 배움
→ 자연스럽게 과확신이 줄어듦
```

**문헌 근거**:
- "Mixup으로 학습한 모델은 calibration이 크게 개선됨" (NeurIPS)
- "label smoothing 효과 + data augmentation 효과를 동시에 얻음"
- "과적합과 과확신을 동시에 억제"

**v8에 추가하면**:
- 과확신 오류 (99% 확신으로 틀림) → 확신도가 낮아져서 LogLoss 폭발 방지
- train/dev 도메인 차이에 더 강건

**주의**: label smoothing과 동시에 사용하면 안 됨 (효과 충돌)

### 2. 밝기 Augmentation 강화

**원리**: train 이미지를 일부러 어둡게 만들어서 dev/test 환경과 비슷하게

```
현재: brightness_limit=(-0.35, 0.2) → 어둡게 35%, 밝게 20%
개선: brightness_limit=(-0.5, 0.1)  → 어둡게 50%로 강화, 밝게는 줄임
     + RandomGamma(gamma_limit=(60, 140)) 추가
```

**이유**: train=218, dev=192 밝기 차이가 명확.
어둡게 하는 augmentation을 더 강하게 해서 모델이 어두운 이미지에 익숙해지게.

### 3. Confidence Penalty / Label Smoothing

**원리**: 정답이 1.0이 아니라 0.95로 학습 → 99% 이상 확신 못 하게

```python
# Label smoothing 0.05
target = 1.0 * (1 - 0.05) + 0.0 * 0.05 = 0.95
→ 모델이 최대 95% 정도까지만 확신하게 됨
→ 틀려도 "95% 확신으로 틀림"이므로 LogLoss가 덜 폭발
```

**주의**: v9에서 label smoothing을 시도했지만 LB 악화.
→ 그때는 focal loss + KD alpha 등과 동시에 바꿔서 뭐가 나쁜지 불분리.
→ 이번에는 label smoothing만 단독으로 테스트 필요.

## 추천 조합

**v8 + Mixup + 밝기 augmentation 강화** (2가지만 변경)

- Mixup: alpha=0.2 (약한 mixup, 과확신 억제)
- 밝기: brightness_limit=(-0.5, 0.1)
- label smoothing: 사용 안 함 (mixup과 충돌)
- 나머지 v8 전부 그대로

## Sources

- [On Mixup Training: Improved Calibration (NeurIPS)](https://arxiv.org/abs/1905.11001)
- [Beyond Overconfidence: Calibration in Neural Networks (2025)](https://arxiv.org/html/2506.09593v1)
- [Calibration in Deep Learning Survey](https://arxiv.org/pdf/2308.01222)
- [Brightness Augmentation for Traffic Sign Classification (99.1%)](https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc)
- [Robust Classification: Data Mollification with Label Smoothing (2024)](https://arxiv.org/html/2406.01494v1)

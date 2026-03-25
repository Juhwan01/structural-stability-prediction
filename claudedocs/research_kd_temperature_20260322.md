# Research: KD Temperature 최적값

> Date: 2026-03-22

## 문제

Video Swin Teacher의 logit 범위가 [-3.14, 3.43]으로 좁음.
T=5.0에서 soft label이 0.35~0.66으로 압축되어 신호가 너무 약함.

## 실제 데이터 기반 시뮬레이션

| T | stable soft label | unstable soft label | 판단 |
|---|-------------------|--------------------|----|
| 1.0 | 0.04~0.06 | 0.05~0.97 | 거의 hard label, KD 의미 없음 |
| **1.5** | **0.11~0.13** | **0.12~0.91** | **적절한 softening** |
| **2.0** | **0.17~0.20** | **0.18~0.85** | **좋은 범위** |
| 3.0 | 0.26~0.28 | 0.27~0.76 | 조금 soft |
| 5.0 | 0.35~0.36 | 0.36~0.67 | 너무 soft, 신호 상실 |

## 문헌 근거

- Hinton (2015): T=1~20 실험, 작은 student에서는 낮은 T가 좋음
- 모델 capacity가 작을 때 T=2.5~4.0이 최적 (Hinton 실험)
- 최근 연구: logit 범위에 비례하여 T를 설정해야 함

## 우리 상황의 특수성

- Teacher logit std = 3.05 → logit 범위가 매우 좁음
- Binary classification (sigmoid) → softmax KD와 다름
- T=5.0이면 logit/T = [-0.63, 0.69] → sigmoid 선형 구간에 빠짐

## 추천

**T=1.5 또는 T=2.0**

- T=1.5: stable=0.12, unstable=0.89 → 충분한 contrast + 약간의 softening
- T=2.0: stable=0.18, unstable=0.82 → 더 많은 softening

## Sources

- [Hinton et al., 2015 - Distilling the Knowledge](https://arxiv.org/pdf/1503.02531)
- [Logit Standardization in KD (CVPR 2024)](https://arxiv.org/html/2403.01427v1)
- [Unified Revisit of Temperature (2025)](https://arxiv.org/abs/2603.02430)
- [Dynamic Temperature KD (2024)](https://arxiv.org/abs/2404.12711)

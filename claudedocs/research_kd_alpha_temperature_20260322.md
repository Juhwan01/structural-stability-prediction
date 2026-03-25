# Research: KD Alpha + Temperature 최적값

> Date: 2026-03-22

## 우리 상황

- Cross-modal KD: Video Teacher → Image Student (서로 다른 modality)
- Teacher logit 범위: [-3.14, 3.43] (좁음)
- Binary classification (sigmoid BCE, softmax KD와 다름)
- 데이터: 1,100장 (소규모)

## 문헌 결과 종합

### Alpha (soft label vs hard label 비중)

| 출처 | 설정 | 비고 |
|------|------|------|
| Keras 공식 가이드 | alpha=0.1 (hard 10%, soft 90%) | soft label 위주 |
| Hinton (2015) | 일반적으로 0.5 | 균형 |
| ICCV 2019 실험 | **alpha=0.25, T=1 → 86% (최고)** | hard label 위주가 좋았음 |
| ImageNet 실험 | alpha=0.9, T=4 | 대규모에서는 soft 위주 |
| Cross-modal KD (CVPR 2024) | 필터링 전략 사용 | modality gap이 크면 soft label 신뢰도 낮음 |

**핵심 발견**: ICCV 2019 "On the Efficacy of Knowledge Distillation"에서
**alpha=0.25, T=1이 가장 좋았다**는 결과가 있음.

### Temperature

| 규모 | 최적 T | 비고 |
|------|-------|------|
| Student가 큰 경우 | T=8+ | 큰 모델은 rich한 정보 흡수 가능 |
| **Student가 작은 경우** | **T=2.5~4.0** | 작은 모델은 너무 soft하면 못 배움 |
| Logit 범위가 좁은 경우 | **T=1~2** | logit/T가 sigmoid 선형 구간에 빠지면 안 됨 |

### Cross-Modal 특수성 (CVPR 2024 C2KD)

**modality gap이 크면 soft label을 맹신하면 안 됨.**
- Video → Image는 modality가 완전히 다름
- Teacher가 틀린 샘플의 soft label이 노이즈가 됨
- **alpha를 낮춰서 hard label 비중을 높이는 게 안전**

## 추천

### 우리 상황에 맞는 최적 설정

**T=1.0, alpha=0.3**

이유:
1. Teacher logit 범위가 -3~3으로 좁음 → T=1.0이면 soft label 범위 0.04~0.97 (적절)
2. Cross-modal gap이 큼 → alpha 낮춰서 hard label 위주
3. ICCV 2019 실험에서 alpha=0.25, T=1이 최고 성능
4. 소규모 데이터 → 안정적 설정 우선

### 비교

| | v8 (이전) | v11 T=5 (실패) | v11 T=1.5 (진행중) | **추천** |
|---|---|---|---|---|
| T | 3.0 | 5.0 | 1.5 | **1.0** |
| alpha | 0.7 | 0.7 | 0.7 | **0.3** |
| soft label 범위 | 0.04~0.99 (R3D) | 0.35~0.66 | 0.12~0.91 | **0.05~0.97** |
| Fold 0 결과 | 0.0048 | 0.2101 | 0.0792 | ? |

## Sources

- [On the Efficacy of KD (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.pdf) - alpha=0.25, T=1 best
- [C2KD: Cross-Modal KD (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.pdf)
- [Unified Revisit of Temperature (2025)](https://arxiv.org/abs/2603.02430)
- [Rethinking Soft Labels: Bias-Variance Tradeoff](https://ar5iv.labs.arxiv.org/html/2102.00650)
- [Hinton et al., 2015](https://arxiv.org/pdf/1503.02531)

# DROID held-out: baseline / Model-FM v1.2 / Model-FM v1.3

## 결론

Open-loop action 오차는 baseline < Model-FM v1.3 < Model-FM v1.2 순이다. v1.3은 v1.2 대비 MSE 14.0%, MAE 4.2% 개선됐지만 baseline 대비 MSE 14.8%, MAE 15.1% 높다. 세 평가 seed 모두 같은 순위다. 실제 task 성공률이나 수행 시간의 비교는 아니다.

## 평가 조건과 집계

세 evaluation.json의 test 경로, trajectory ID 0~59, seed 42/43/44, execution horizon 8, denoising steps 4, 요청 steps 100000, dataset_manifest_sha256이 같다. 9개 로그 모두 중복 없는 60개 trajectory 결과와 Done이 있으며 trajectory별 처리 frame 수도 같다. manifest hash 일치는 보고된 manifest 기준이며 영상/parquet 전체의 독립 hash 검증은 아니다.

각 로그의 `MSE for trajectory ...` 결과를 사용해 episode 60개를 동일 가중치로 평균하고, 평가 seed 3개를 동일 가중치로 평균했다. Frame 수로 가중한 평균이 아니다. 로그의 action 배열은 8차원이며 joint 7개와 gripper 1개를 함께 계산한 unnormalized 오차다. 순수 joint rad/degree 지표가 아니다. Seed는 평가 반복이며 독립 학습 반복을 의미하지 않는다.

## 평균 성능 (낮을수록 좋음)

| 모델 | MSE | MAE | baseline 대비 MSE | baseline 대비 MAE |
|---|---:|---:|---:|---:|
| baseline | 0.009079 | 0.052202 | — | — |
| Model-FM v1.2 | 0.012126 | 0.062723 | +33.6% | +20.2% |
| Model-FM v1.3 | 0.010427 | 0.060109 | +14.8% | +15.1% |

## seed별 성능

| seed | baseline MSE | v1.2 MSE | v1.3 MSE | baseline MAE | v1.2 MAE | v1.3 MAE |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.009010 | 0.012006 | 0.010226 | 0.052222 | 0.062467 | 0.059751 |
| 43 | 0.009174 | 0.012093 | 0.010502 | 0.052288 | 0.062828 | 0.060461 |
| 44 | 0.009054 | 0.012278 | 0.010553 | 0.052096 | 0.062873 | 0.060115 |

## episode별 개선 수

각 episode의 세 seed 평균을 비교했다. 전체 수치는 `held_out_comparison.csv`에 있다.

| 비교 | MSE 감소 episode | MAE 감소 episode |
|---|---:|---:|
| v1.2 vs baseline | 12/60 | 7/60 |
| v1.3 vs baseline | 16/60 | 8/60 |
| v1.3 vs v1.2 | 44/60 | 44/60 |

v1.3의 v1.2 대비 개선은 다수 episode에서 나타나지만 baseline 대비 MAE는 52/60 episode에서 높다. 이 수치는 통계적 유의성 검정 결과가 아니다.

## 해석과 한계

Open-loop는 시연의 실제 관측/state를 입력한다. 모델 action이 바꾼 다음 관측으로 오차가 누적되는 closed-loop 과정과 task 성공률은 측정하지 않는다. 각 모델 processor를 사용한 비교이므로 순수 LoRA weight 효과만 분리하지 않는다. 결과만으로 학습 설정 변경의 인과 효과나 과적합, 학습 부족, 구현 결함을 확정할 수 없다.

기존 분석에서 확인한 upstream plot 저장 동작은 같은 seed PNG를 episode마다 덮어쓴다. 각 seed PNG는 마지막 trajectory 59의 결과이므로 전체 성능 비교에는 60개 episode 로그를 사용한다. 현재 로그에는 joint/gripper별 오차가 없어 어느 action 차원이 차이를 주도하는지 분해할 수 없다.

## 다음 검증

1. joint/gripper별 MAE와 gripper 전환 구간 오차를 저장하고 trajectory별 plot을 보존한다.
2. 학습 전 LoRA B=0 및 merge export가 baseline과 같은 processor/input/seed에서 같은 출력을 내는지 GPU에서 검사한다.
3. processor 통계, 학습 설정, 중간 checkpoint를 확보해 동일 조건으로 비교한다.
4. 실제 동작 성능은 동일 task의 closed-loop 성공률과 수행 시간으로 별도 평가한다.

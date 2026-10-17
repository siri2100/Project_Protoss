# Project Protoss v1.6 — GR00T baseline / Model-FM/S/E 학습·평가

네 모델은 분할하지 않은 같은 원본 DROID 데이터 전체을 사용한다. baseline은 원본 pretrained 평가 전용이며, Model-FM/S/E는 같은 pretrained 모델에서 독립적으로 LoRA 학습한다.

| 모델 | 학습 | Loss | 구현 상태 |
|---|---|---|---|
| baseline | 없음 | 없음 | 원본 평가 가능 |
| Model-FM | LoRA | FM | 학습·평가 실행 경로 제공 |
| Model-S | LoRA | FM (현재 Model-FM과 동일) | 학습·평가 실행 가능 |
| Model-E | LoRA | FM (현재 Model-FM과 동일) | 학습·평가 실행 가능 |

현재 Model-FM/S/E는 같은 FM loss, 데이터, LoRA 설정으로 독립 학습한다. Model-S/E는 향후 보조 loss를 추가하기 위한 실험 이름이며 현재 smoothness/efficiency loss는 적용하지 않는다. 같은 seed와 설정에서는 같은 결과가 나올 수 있고, 이 단계의 차이를 동작 특성 개선으로 해석하지 않는다.

## 사용 문서

| 목적 | 문서 |
|---|---|
| 설치·데이터 준비·LoRA 학습·재개·Hugging Face 업로드 | [readme_train.md](readme_train.md) |
| 모델 다운로드·추론·open-loop 평가·RoboLab 평가 | [readme_test.md](readme_test.md) |
| 출력 동등성·processor 차이·checkpoint 추세 진단 | [README_analysis.md](README_analysis.md) |

처음 학습한다면 학습 문서부터, 이미 학습된 모델을 평가한다면 추론·평가 문서부터 진행한다.

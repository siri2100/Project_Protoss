# Project Protoss

현재 v1.2는 GR00T-N1.7-DROID 기반 네 모델을 같은 원본 DROID train/test 데이터로 비교한다.

| 모델 | 학습 | Loss |
|---|---|---|
| baseline | 없음, pretrained 평가 | 없음 |
| Model-FM | LoRA | FM |
| Model-S | LoRA | FM (현재 Model-FM과 동일) |
| Model-E | LoRA | FM (현재 Model-FM과 동일) |

설치부터 데이터 준비·학습·평가·Hugging Face 보관까지는 [README.md](../README.md)를 따른다. Model-FM/S/E는 현재 같은 FM loss로 각각 학습하며 향후 보조 loss를 추가할 수 있다.

- 공통 데이터 준비: `prepare_groot_dataset.py`
- Model-FM/S/E LoRA 학습: `train_groot_models.py --model FM|S|E`
- 네 모델 공통 평가: `eval_groot_models.py --model baseline|FM|S|E`

과거 버전 설명은 [v1.0](../README_v1.0.md), [v1.1](../README_v1.1.md)에 보관한다.

CLI는 이 폴더에, 구현은 `src/`, 테스트는 `src/tests/`에 있다. 서버에 복사할 때 `src/`도 함께 포함한다.

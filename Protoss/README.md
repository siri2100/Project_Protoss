# Project Protoss

현재 v1.5는 GR00T-N1.7-DROID 기반 네 모델을 분할하지 않은 같은 원본 DROID 데이터 전체로 비교한다.

| 모델 | 학습 | Loss |
|---|---|---|
| baseline | 없음, pretrained 평가 | 없음 |
| Model-FM | LoRA | FM |
| Model-S | LoRA | FM (현재 Model-FM과 동일) |
| Model-E | LoRA | FM (현재 Model-FM과 동일) |

설치부터 데이터 준비·학습·평가·Hugging Face 보관까지는 [README.md](../README.md)를 따른다. Model-FM/S/E는 현재 같은 FM loss로 각각 학습하며 향후 보조 loss를 추가할 수 있다.

- 공통 데이터 준비: `prepare_groot_dataset.py`
- Model-FM/S/E LoRA 학습: `train.py --model FM|S|E`
- 네 모델 공통 평가: `eval_groot_models.py --model baseline|FM|S|E`

과거 버전 설명은 [v1.0](../README_v1.0.md), [v1.1](../README_v1.1.md)에 보관한다.

`train.py`의 `main()`에 실제 학습 반복문이 있다. 각 optimizer step에서 batch 전처리 결과를 GPU로 옮기고, 모델 forward로 loss를 계산하고, gradient accumulation 후 가중치를 갱신한다. 이미지·state·action 전처리는 기존 GR00T dataset/collator가 수행하므로 루프에서 중복 적용하지 않는다.

`src/groot_lora.py`는 PEFT의 `LoraConfig`와 `get_peft_model()`로 DiT에서 선택한 Linear에 LoRA를 적용한다(현재 `to_q`). 원본 모델은 동결하고 LoRA A/B만 학습한다. 학습 종료 시에는 adapter와 processor만 저장하고, 추론 시 원본 pretrained 모델과 adapter를 로드하여 PEFT `merge_and_unload()`로 병합한다. PEFT adapter는 `adapter/`에 저장하며, 이전 자체 구현 checkpoint에서 PEFT 학습으로 resume하는 것은 지원하지 않는다.

`src/training_runtime.py`는 모델·데이터 준비, optimizer/scheduler, checkpoint/resume, adapter 저장을 담당한다. 현재 학습 루프는 CUDA GPU 한 개를 사용한다. Checkpoint는 adapter와 Trainer 형식의 optimizer·scheduler·RNG 상태를 저장하며, resume 시 데이터는 기존 GR00T 방식대로 global step에 따라 reseed한다. 중단 전과 동일한 데이터 순서로 재개하는 방식은 아니다.

나머지 CLI는 이 폴더에, 공통 구현은 `src/`, 테스트는 `src/tests/`에 있다. 서버에 복사할 때 `src/`도 함께 포함한다.

v1.3: 영상은 현재 프레임 `[0]`, state dropout은 기본 `0.0`이다. 기존 데이터는 재사용하고 새 checkpoint 폴더에서 재학습한다.

학습 완료 시 모델 output 폴더에 `training_loss.png`와 원본 수치 `training_loss.csv`를 저장한다. X축은 optimizer step, Y축은 gradient accumulation batch들의 평균 flow-matching loss다. Loss 기록은 checkpoint에도 포함되어 `--resume` 시 이어진다. 이 기능 이전 checkpoint에 loss 기록이 없다면 재개 이후 기록만 표시한다.

서버·평가의 `--model-path`에는 학습 output 폴더를 지정한다. 기존 PEFT output도 `adapter/`, `processor/`, `lora_settings.json`이 있으면 지원한다. 원본 pretrained 모델 경로를 바꾸려면 `--base-model-path`를 추가한다. 새 output의 `inference/`에는 모델 weight 대신 `adapter_inference.json`만 저장되므로 adapter·processor 폴더도 함께 보관한다.

서버는 `serve.py`로 실행한다. 서버·평가 실행기는 별도 GR00T checkout에 `patches/gr00t_adapter_inference.patch`를 자동 적용한다. 배포할 때 `src/`와 `patches/`를 함께 포함한다.

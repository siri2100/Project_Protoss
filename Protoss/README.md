# Project Protoss

현재 v1.7은 GR00T-N1.7-DROID 기반 네 모델을 분할하지 않은 같은 원본 DROID 데이터 전체로 비교한다.

| 모델 | 학습 | Loss |
|---|---|---|
| baseline | 없음, pretrained 평가 | 없음 |
| Model-FM | LoRA | FM |
| Model-S | LoRA | FMD + normalized joint jerk |
| Model-E | LoRA | FMD + joint path length |

설치·데이터 준비·학습·업로드는 [readme_train.md](../readme_train.md), 모델 다운로드·추론·평가는 [readme_test.md](../readme_test.md)를 따른다. Model-FM(v1.6.14)은 FM 기준 모델이며 Model-S(v1.7)는 FMD + 정규화 jerk, Model-E는 FMD + 관절 경로 길이로 독립 학습한다.

- 공통 데이터 준비: `prepare_groot_dataset.py`
- Model-FM/S/E LoRA 학습: `train.py --model FM|S|E`
- 네 모델 공통 평가: `eval_groot_models.py --model baseline|FM|S|E`

과거 버전 설명은 [v1.0](../README_v1.0.md), [v1.1](../README_v1.1.md)에 보관한다.

`train.py`의 `main()`에 실제 학습 반복문이 있다. 각 optimizer step에서 batch 전처리 결과를 GPU로 옮기고, 모델 forward로 loss를 계산하고, gradient accumulation 후 가중치를 갱신한다. 이미지·state·action 전처리는 기존 GR00T dataset/collator가 수행하므로 루프에서 중복 적용하지 않는다.

`src/groot_lora.py`는 PEFT의 `LoraConfig`와 `get_peft_model()`로 DiT의 모든 block의 `to_q`, `to_k`, `to_v`에만 LoRA를 적용한다. 원본 모델은 동결하고 LoRA A/B만 학습한다. 학습 종료 시에는 adapter와 processor만 저장하고, 추론 시 원본 pretrained 모델과 adapter를 로드한다. 기본 `unmerged`는 병합 없이 LoRA를 적용하고, `merged`는 PEFT `merge_and_unload()`로 병합한다. PEFT adapter는 `adapter/`에 저장하며, 이전 자체 구현 checkpoint에서 PEFT 학습으로 resume하는 것은 지원하지 않는다.

`src/training_runtime.py`는 모델·데이터 준비, optimizer/scheduler, checkpoint/resume, adapter 저장을 담당한다. 현재 학습 루프는 CUDA GPU 한 개를 사용한다. Checkpoint는 adapter와 Trainer 형식의 optimizer·scheduler·RNG 상태를 저장하며, resume 시 데이터는 기존 GR00T 방식대로 global step에 따라 reseed한다. 중단 전과 동일한 데이터 순서로 재개하는 방식은 아니다.

나머지 CLI는 이 폴더에, 공통 구현은 `src/`, 테스트는 `src/tests/`에 있다. 서버에 복사할 때 `src/`도 함께 포함한다.

현재 학습은 원본 processor를 override 없이 로드하고 normalization 통계를 덮어쓰지 않으며, 원본 영상 시점을 유지한다. 저장되는 processor JSON은 원본과 동일한 파일이다. 기존 `[0]`/subset normalization 실험은 재개하지 않고, 기존 데이터로 새 output에서 pretrained 모델부터 LoRA만 다시 학습한다.

학습 완료 시 모델 output 폴더에 `training_loss.png`와 원본 수치 `training_loss.csv`를 저장한다. X축은 optimizer step, Y축은 gradient accumulation batch들의 평균 flow-matching loss다. Loss 기록은 checkpoint에도 포함되어 `--resume` 시 이어진다. 이 기능 이전 checkpoint에 loss 기록이 없다면 재개 이후 기록만 표시한다.

서버·평가의 `--model-path`에는 학습 output 폴더를 지정한다. 기존 PEFT output도 `adapter/`, `processor/`, `lora_settings.json`이 있으면 지원한다. 원본 pretrained 모델 경로를 바꾸려면 `--base-model-path`를 추가한다. 새 output의 `inference/`에는 모델 weight 대신 `adapter_inference.json`만 저장되므로 adapter·processor 폴더도 함께 보관한다.

서버는 `serve.py`로 실행한다. 서버·평가 실행기는 별도 GR00T checkout에 `patches/gr00t_adapter_inference.patch`를 자동 적용한다. 배포할 때 `src/`와 `patches/`를 함께 포함한다.

RoboLab이 instruction을 필드 mapping으로 전달하면 `src/robolab_gr00t_client.py`는 `default` 문자열을 선택해 GR00T에 보낸다. 클라이언트 수정 후 simulator 서버에서 이 파일을 `/workspace/RoboLab/policies/gr00t/client.py`로 복사하고 평가 프로세스를 재시작한다.

중간 step을 추론하려면 서버 명령에 `--checkpoint-step 500`을 추가하고 `--model-path`는 run 루트로 유지한다. Hugging Face를 사용하면 run 다운로드 후 `checkpoint-500/adapter`를 선택한다. Processor는 해당 checkpoint 것을 우선하고 없으면 같은 run의 것을 사용한다. `--base-model-path`로 현재 서버의 pretrained 경로를 지정한다. 지정 step이 없으면 오류로 종료하며 최종 adapter로 대체하지 않는다. 최대 3개 checkpoint 보관 설정으로 삭제된 step은 남아 있는 별도 파일 없이는 복구할 수 없다.

추론 디버그 모드: `--lora-mode pretrained`는 원본 모델·원본 processor, `disabled`는 원본 모델·학습 processor, `unmerged`는 LoRA를 병합 없이 적용, `merged`는 LoRA를 병합하여 적용한다. 병합을 끄는 것은 LoRA를 끄는 것과 다르다.

`lora_settings.json`의 `processor_policy=pretrained-v1`과 processor SHA256으로 새 학습을 구분한다. 데이터셋 생성 후 원본 normalization 통계가 바뀌면 학습을 중단한다.

모델 저장 위치는 프로젝트 바로 상위의 `checkpoint/`다. 예: `/workspace/Project_Protoss`의 모델은 `/workspace/checkpoint`에 저장한다. 학습 문서의 `CHECKPOINT_ROOT` 설정으로 원본 pretrained와 학습 run을 저장하고, 완료된 run 전체를 Hugging Face private 모델 저장소에 업로드한다. 추론 서버에서는 추론 문서에 따라 원본 pretrained와 업로드한 run을 Hugging Face에서 같은 `checkpoint/` 아래로 내려받고, `--model-path`에 다운로드한 run을, `--base-model-path`에 원본 pretrained 경로를 지정한다.

Model-S는 `loss = fm_weight × FMD + jerk_lambda × normalized_jerk_loss`를 사용한다. FMD는 같은 관측·noise·timestep에서 pretrained와 pretrained+LoRA의 flow velocity 차이에 대한 masked MSE다. Teacher는 PEFT adapter를 끄고 gradient 없이 실행하며 student에만 LoRA를 적용한다. Model-E도 같은 teacher/student 경로를 사용하며 Model-FM은 정답 action FM을 사용한다.

Student clean endpoint의 관절 7개를 pretrained 통계로 역정규화한 뒤 `jerk = Δ³q/dt³`를 계산한다. 네 점이 모두 유효한 window만 평균하며 gripper·EEF·padding을 제외한다. `normalized_jerk_loss = mean(jerk²)/jerk_ref²`이고 기준 RMS는 정답 10개 batch에서 구해 고정한다. 최소 기준은 `1 rad/s³`이며 준비 단계는 optimizer step을 사용하지 않는다. 설정은 `ModelSConfig`, 계산은 `masked_jerk_loss()`에서 수정한다.

`--s-fmd-weight`(기존 `--s-fm-weight`도 지원), `--s-jerk-lambda`, `--s-jerk-reference`, `--s-calibration-batches`, `--s-dt`로 실험을 조정한다. 기본 FMD 가중치는 1, jerk lambda는 0.01이다. 기준 RMS를 직접 지정하면 측정을 생략한다. 기존 velocity 옵션은 사용하지 않는다.

콘솔과 CSV에 FMD·가중 FMD·jerk·total을 기록하고 그래프는 FMD/정규화 jerk/total 세 패널이다. Raw jerk와 기준 RMS도 저장한다. 이전 FMD+velocity checkpoint를 새 목적 함수로 재개하지 않고 새 output에서 학습한다. 실제 rollout의 jerk와 성공률 개선은 별도 검증한다.

Model-E는 `src/model_e_loss.py`에서 `loss = fm_weight × FMD + path_length_weight × path_length_loss`로 계산한다. `path_length_loss`는 student clean endpoint를 pretrained 통계로 역정규화한 관절 7개의 `sum_t ||q[t+1]-q[t]||₂`에 대한 batch 평균이다. Gripper·EEF·padding·결측 구간을 제외하며 정규화나 dt 계수를 적용하지 않는다. `--e-fmd-weight`(기본 1), `--e-path-length-weight`(기본 0.01)로 두 항을 조절한다. 콘솔·CSV·그래프에 FMD/path length/total을 기록하고 loss 설정도 checkpoint에 저장한다. 기존 FM-only Model-E는 새 output에서 다시 학습한다.

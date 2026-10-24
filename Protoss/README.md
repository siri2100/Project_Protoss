# Project Protoss

GR00T-N1.7-DROID의 pretrained baseline과 Model-S/E를 비교한다. 설치·학습은 [readme_train.md](../readme_train.md), 추론·평가는 [readme_test.md](../readme_test.md)를 따른다.

| 모델 | Loss |
|---|---|
| baseline | 학습 없이 pretrained 평가 |
| Model-S | `fm_weight × FMD + sparc_weight × (-mean joint SPARC)` |
| Model-E | `fm_weight × FMD + fast_weight × fast_loss + limit_weight × limit_loss` |

`run.py`가 공통 진입점이며, 모델 이름·step·loss weight 등은 CLI 인자로, 기본값은 `run_config.py`로 조절한다. 현재 v1.7.13은 0부터 세는 DiT 짝수 block(0, 2, 4, …)의 QKV·FFN Linear에 LoRA를 적용한다.

| 코드 | 역할 |
|---|---|
| `train.py`, `src/training_runtime.py` | S/E 학습 루프, checkpoint 저장·재개 |
| `src/model_s_forward.py` | 같은 관측·noise·timestep에서 teacher/student FMD |
| `src/model_s_loss.py`, `src/sparc_loss.py` | 관절 속도 프로파일의 SPARC loss |
| `src/model_e_loss.py` | teacher 관절·그리퍼 궤적의 시간 압축 추종, 관절 속도·가속도 초과 penalty |
| `serve.py`, `src/inference_loading.py` | pretrained·adapter·processor 로딩 |
| `evaluate_robolab.py`, `src/training_evaluation.py` | 학습 종료 후 서버·평가·점수 업로드 |
| `src/robolab_results.py` | task별 분리, metric·성공률 집계, HF 업로드 |
| `src/setup_recipes.py`, `src/experiment_recipes.py` | `run.py` 명령 구현 |

모델은 프로젝트 상위 `checkpoint/`에 저장한다. 학습은 원본 DROID 데이터 전체와 pretrained processor 통계를 사용하며, LoRA adapter·processor·설정·loss CSV/그래프를 저장한다. 재개 시 optimizer·scheduler·RNG 상태를 복원한다. Loss나 LoRA 대상이 다른 예전 checkpoint로 재개할 수 없다.

학습 → HF 업로드 → 학습 프로세스 종료 → 추론 서버 → RoboLab 평가 → 점수 HF 업로드 순으로 진행한다. 기본 평가는 지정한 5개 task를 2 env × 5회, task당 10 episode씩 실행한다. 기존 결과는 보관하고 매번 새로 평가한다.

추론은 `unmerged`(기본), `merged`, `disabled`, `pretrained` 모드를 지원한다. 중간 checkpoint는 `--checkpoint-step`으로 선택한다. `setup_robolab_595.sh`와 `patches/`는 현재 Isaac Sim 6.0.1 / 드라이버 595 환경에 필요한 호환 코드이므로 함께 배포한다. 내부 CLI는 `python -m src.<모듈>`로 실행한다. 테스트는 `src/tests/`에 있다.

Model-E는 같은 noise·timestep에서 얻은 teacher의 clean endpoint 추정 궤적을 고정하고 `q_fast[k] = q_teacher[min(speedup*k, H-1)]`로 선형 보간한다. 기본 `speedup=1.2`이며 마지막 위치는 유지한다. 실제 denoising rollout 전체가 아닌 학습 중 추정 궤적에 적용하는 surrogate다. 원본 학습 데이터의 시간축은 바꾸지 않는다.

관절·그리퍼를 각자의 pretrained 통계로 역정규화한 뒤 보간하고, horizon 전체에서 고정된 채널별 scale로 나눈 MSE를 계산한다. 관절 평균 MSE와 그리퍼 평균 MSE의 합이 `fast_loss`다. Padding·결측을 넘어서 보간하지 않으며 연속 위치가 최소 3개인 구간만 사용한다. Teacher에는 gradient를 전달하지 않는다.

`limit_loss`는 `mean(relu(abs(velocity)/v_limit - 1)^2) + mean(relu(abs(acceleration)/a_limit - 1)^2)`다. 관절 7개에 적용하고 현재 raw 관절 위치 → 첫 action의 속도도 포함한다. 기본 상한 2 rad/s·10 rad/s²는 조절 가능한 실험용 soft bound이며 실제 controller 제한을 대신하지 않는다. 기존 path-length checkpoint는 새 목적함수로 재개할 수 없다. Model-E 콘솔·CSV·그래프에는 FMD, fast, limit, total을 기록한다.

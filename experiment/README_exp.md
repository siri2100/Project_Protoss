# RoboLab 평가 결과 정리

현재 실험은 `v1.6.10 : all blocks ffn`으로, 모든 DiT block의 FFN Linear에 LoRA를 적용한다. 원본 pretrained 가중치는 고정하고 LoRA만 학습한다.

아래 명령은 Mac에서 다운로드한 결과 ZIP을 풀고, task별 JSONL 분리와 metric 평균 계산을 수행한다. `NAME`과 `STEP`은 실제 결과 파일명에 맞게 수정한다. ZIP에는 `${NAME}_model_fm_step${STEP}/episode_results.jsonl` 구조가 있어야 한다.

```bash
(
set -e
NAME=v1.6.10
STEP=500
EXP_ROOT="$HOME/Desktop/Project_Protoss/experiment/RoboLab"
RUN_NAME="${NAME}_model_fm_step${STEP}"

# 결과 ZIP 이동 및 압축 해제
mkdir -p "$EXP_ROOT"
mv -i "$HOME/Downloads/$RUN_NAME.zip" "$EXP_ROOT/"
cd "$EXP_ROOT"
unzip -n "$RUN_NAME.zip"
test -f "$RUN_NAME/episode_results.jsonl"

# task별 분리: 입력에서 처음 등장한 task부터 RoboLab05, 06, ...으로 저장
python3 split_episode_results.py "$RUN_NAME/episode_results.jsonl" 5 \
  --overwrite --suffix .jsonl

# 첫 task의 metric 평균 계산
python3 calculate_metric_averages.py "$RUN_NAME/RoboLab05.jsonl"
)
```

분리 결과는 해당 실험 폴더에 `RoboLab05.jsonl`, `RoboLab06.jsonl`, …으로 저장된다. 번호는 고정 task ID가 아니라 입력의 task 등장 순서이므로, 분리 스크립트 출력에서 task 이름을 확인한다. 다른 task의 평균은 마지막 명령의 파일명을 바꿔 계산한다. `--overwrite`는 기존 분리 JSONL을 덮어쓴다.

ZIP을 이미 이동했다면 `mv` 줄을 생략한다. 압축 해제까지 끝났다면 분리·평균 계산 명령만 실행한다.

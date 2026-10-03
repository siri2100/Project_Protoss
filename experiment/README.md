# RoboLab 평가방법
```bash
cd /experiment/RoboLab

# 파일 분리
python3 split_episode_results.py episode_results.jsonl 1 --overwrite --suffix _GR00T-N1.6-DROID.jsonl

# 평균값 계산
python3 calculate_metric_averages.py RoboLab01_GR00T-N1.7-DROID.jsonl
```
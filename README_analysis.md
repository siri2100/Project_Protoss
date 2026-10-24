# 모델 진단

현재 Model-S/E의 학습·추론·평가는 [readme_train.md](readme_train.md)와 [readme_test.md](readme_test.md)를 따른다.

예전 Model-FM의 held-out 분석·전체 모델 export 도구는 현재 학습 경로에서 사용하지 않아 제거했다. 이전 구현과 분석 절차는 Git 이력에 남아 있다.

현재 진단은 `python3 run.py check`, `python3 run.py plot-distribution --name 모델이름`, 서버의 `--lora-mode` 비교를 사용한다. 명령은 `Protoss` 폴더에서 실행한다.

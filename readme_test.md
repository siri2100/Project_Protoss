# Project Protoss - Test

### Step 01. Set environment and Download Robolab
Follow README_train.md

### Step 02. Evaluate model using one terminal

```bash
python3 run.py evaluate --name v1.7.11_Model-E
```

### Step 03. Evaluate model using dual terminal(server, client)

```bash
python3 run.py server --name v1713_model_s_step500 --checkpoint-step 500
```

```bash
python3 run.py server-baseline
```

```bash
python3 run.py client --name baseline
```

### Step 04. Evaluate model with one task

```bash
python3 run.py client --name v1713_model_s_step500 --task BananaOnPlateTask --num-envs 1 --num-runs 1
```

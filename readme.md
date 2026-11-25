# IAEML Assignments

A JAX driving environment and a PPO agent for IAEML course Assigment. In the 48 × 26 Service Hall, the taxi must turn around two wall sections and avoid moving obstacles. The planned route is about 58 units long. The trained policy reached the goal in 111 of 128 held-out episodes.

| Assignment | Report | Video |
| --- | --- | --- |
| 1. Driving environment | [PDF](reports/assignment_1%20report.pdf) | [Scripted driving demo](artifacts/room/assignment1_environment.mp4) |
| 2. Reinforcement learning | [PDF](reports/assignment_2%20report.pdf) | [Trained PPO demo](artifacts/room/assignment2_ppo.mp4) |

[Model checkpoint](artifacts/room/policy.npz) · [Evaluation results](artifacts/room/evaluation.csv) · [Training log](artifacts/room/training.csv)

## Run locally

From the repository root in PowerShell, using Python 3.14:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/demo_driving.py --controller manual --nominal
.\.venv\Scripts\python.exe src/demo_driving.py --controller ppo
.\.venv\Scripts\python.exe src/train_driving.py evaluate
```

In VS Code, select **Terminal → Run Task** to drive, replay, train, or evaluate.
Use the arrow keys or WASD to drive, **R** to reset, **Space** to pause, and **Esc** to exit.

## Reproduce the recorded episodes

```powershell
.\.venv\Scripts\python.exe src/demo_driving.py --controller route --seed 100003 --video artifacts/room/assignment1_environment.mp4
.\.venv\Scripts\python.exe src/demo_driving.py --controller ppo --seed 200001 --video artifacts/room/assignment2_ppo.mp4
```

The included checkpoint reproduces the PPO replay and the evaluation on seeds 200000–200127. The logged run started from pretrained weights. To train a new policy from random weights, use the following commands. The new run writes its outputs to `artifacts/room/fresh-training/`.

```powershell
.\.venv\Scripts\python.exe -u src/train_driving.py train
.\.venv\Scripts\python.exe src/train_driving.py evaluate --checkpoint artifacts/room/fresh-training/policy.npz --output artifacts/room/fresh-training/evaluation.csv
```

## Source files

- [driving_env.py](src/env/driving_env.py) implements vehicle dynamics, observations, navigation, and rewards.
- [demo_driving.py](src/demo_driving.py) handles interactive driving, replay, and MP4 recording.
- [train_driving.py](src/train_driving.py) trains and evaluates the policy. The [PPO implementation](src/algorithms/ppo.py) batches simulation and optimization with JAX.

The project builds on the [InnoRL course starter](https://github.com/InnoRL/IAEML-assignment) and implements [Schulman et al.'s PPO](https://arxiv.org/abs/1707.06347). The starter demo is available at `python src/main.py`.

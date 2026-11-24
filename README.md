# Trustworthy Skin Lesion Triage

A 7-class dermoscopic skin lesion classifier (HAM10000) built to be **accurate, honest about its
uncertainty, checked for shortcut learning, tested on outside data, audited for subgroup
performance, and explainable** — with a thin Streamlit demo.

> **Disclaimer:** educational research project. Not a medical device and not a diagnosis.

## Status
| Phase | Topic | Status |
|---|---|---|
| 1 | Setup | done |
| 2 | Data pipeline | todo |
| 3 | Models & training | todo |
| 4 | Evaluation | todo |
| 5 | Core experiments & ablations | todo |
| 6 | Foundation models & data efficiency | todo |
| 7 | Calibration, uncertainty, abstention | todo |
| 8 | Ensemble, TTA, threshold tuning | todo |
| 9 | Segmentation & multitask | todo |
| 10 | Robustness & shortcut audit | todo |
| 11 | External validation | todo |
| 12 | Fairness & subgroups | todo |
| 13 | Explainability | todo |
| 14 | Similar-case retrieval | todo |
| 15 | Streamlit app | todo |
| 16 | Documentation | todo |

## Setup
All compute runs on **Kaggle Notebooks** (free GPU; HAM10000 is hosted there), so nothing large
is installed locally.

1. Push this repo to GitHub.
2. On Kaggle: New Notebook → File → Import Notebook → `notebooks/kaggle_runner.ipynb`.
3. Settings: GPU on, Internet on, add dataset `kmader/skin-cancer-mnist-ham10000`
   (and a `GITHUB_TOKEN` secret if the repo is private).
4. Set `REPO_URL` / `BRANCH` / `PHASE_COMMANDS` and Run All. It runs `python -m src.check_env`
   (imports + GPU detection) and `pytest` by default.
5. Download `results.zip` from the notebook Output and unzip into the repo.

Local install is optional (Python 3.12; PyTorch wheels lag behind the newest Python):
`pip install -r requirements.txt && python -m src.check_env && pytest`.
Paths come from `configs/paths.yaml` locally and `configs/paths_kaggle.yaml` on Kaggle
(auto-detected).

## Layout
```
configs/    YAML configs (paths, defaults, experiments) — no hardcoded paths
src/        library code: data, models, train, evaluate, uncertainty, ...
tests/      pytest unit tests (run in CI)
results/    plain-file outputs for every experiment (the only source of reported numbers)
notebooks/  exploration only
app/        Streamlit demo
docs/       report, model/dataset cards, viva notes, guidance.yaml
data/       datasets (git-ignored)
```

## Results
_Filled in from `results/` once experiments have run. No numbers are reported before then._

## Reproduce
_Exact commands added as each phase lands._

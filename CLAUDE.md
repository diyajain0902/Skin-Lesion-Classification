# CLAUDE.md — Trustworthy Skin Lesion Triage

Project rules that persist across sessions. Read before changing anything.

## Theme
"Trustworthy skin lesion triage": a HAM10000 (7-class) classifier that is accurate, honest about
its uncertainty, checked for shortcut learning, tested on outside data, audited for subgroup
performance, and explainable. ML-focused; the Streamlit UI is a thin layer. **No edge/hardware
deployment.** Students must be able to explain every line in a viva: clean, commented code, short
comments on key design decisions.

## Stack
PyTorch, timm, open_clip, segmentation_models_pytorch, albumentations, scikit-learn,
scipy/statsmodels, pandas, matplotlib/seaborn, pytorch-grad-cam, MAPIE or hand-written conformal,
Optuna, MLflow (local file store `mlruns/`), Streamlit, PyYAML, Kaggle API.

## Non-negotiable rules
1. **Split by `lesion_id`, never by image.** Train/val/test = 70/15/15, stratified by class.
   Val is split again by lesion into **val_a** (model selection, temperature scaling, threshold
   tuning) and **val_b** (conformal calibration). Assert zero `lesion_id` overlap between ALL
   splits in code. Run a perceptual-hash near-duplicate check across splits and report it.
2. **Seeds everywhere** (python, numpy, torch, cuda). The SPLIT seed is fixed forever (all
   experiments share identical splits). TRAINING seeds vary: 42, 43, 44. Report mean ± std.
3. **Never invent numbers.** Every number in README/report must come from a file in `results/`.
   If a result does not exist, run the experiment or say so.
4. **Suspiciously high results (e.g. >95% accuracy) → stop and audit for leakage.**
5. Fit normalisation/statistics on **train only**. The **test set is never touched** for
   training, tuning, model selection, calibration, threshold choice or ensemble selection.
   Use val_a / val_b.
6. **Everything configurable via YAML in `configs/`.** No hardcoded paths.
7. Every script has `--debug` (small subset, 1–2 epochs) for smoke testing.
8. Auto-detect CUDA/CPU and adjust batch size. AMP on GPU. Training resumable from checkpoints.
9. **Medical metrics**: macro F1, per-class P/R/F1, one-vs-rest AUROC, PR curves, confusion
   matrix, sensitivity at fixed specificity, melanoma sensitivity & specificity. Bootstrap 95% CIs
   (1000 resamples, **resampled by lesion**) on headline test metrics.
10. Track every run in MLflow **and** write plain files to `results/<experiment_name>/`.
11. **Estimate runtime before any long job.** If compute is limited, ask which experiments to
    prioritise — never silently skip.

## Workflow
- Phases 1–16 in order. Each phase: own git branch (`phase-N-<name>`), run its verification,
  fix problems, print a short summary, commit, merge to `main`. Do not start the next phase until
  the current one passes.
- **Do not add Claude as a co-author/collaborator** in commits (no `Co-Authored-By` trailers).
- If something fails: find the root cause, fix, re-run. Never bypass checks to make them pass.
- Ask briefly when a decision depends on hardware, time or preferences, then continue.

## Environment notes
- Hardware: NVIDIA RTX 3050 Laptop, **4 GB VRAM** → small batches, AMP, possibly grad accumulation.
- **Laptop is code-only (user's explicit wish).** Do NOT pip/uv install packages, download
  datasets, model weights or anything large onto the laptop. All installs, data and training run
  on **Kaggle Notebooks** via `notebooks/kaggle_runner.ipynb` (clones the GitHub repo, installs
  requirements, runs the phase's commands, zips `results/` + `splits/` for download).
  Locally only use stdlib tools (e.g. `python -m py_compile`) for checks. CI runs pytest on GitHub.
- Path selection: `configs/paths_kaggle.yaml` is auto-used when `/kaggle/input` exists;
  `$SKIN_PATHS_CONFIG` overrides; otherwise `configs/paths.yaml`.
- Results flow back by downloading `results.zip` from the Kaggle notebook output and committing
  `results/` (small plain files only). Fixed splits are committed under `splits/`.
- Repo lives in OneDrive: keep large data/checkpoints out of git (see `.gitignore`); the data root
  is configurable in `configs/paths.yaml`.

## Guidance / app rules (Phase 15)
- `docs/guidance.yaml` is hand-written from real, fetched sources (AAD, NHS, Cancer Research UK);
  every class entry names its source. Unsourced facts are left out.
- Never recommend medicines/doses/treatments. Never say "you do not have cancer" / "no need to
  worry". Every recommendation ends in "see a doctor". Guidance is educational, not clinically
  reviewed. Clinical teal theme, no gradients/emojis/dark mode, non-dismissible amber disclaimer.

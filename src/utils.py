"""Shared helpers: config loading, seeding, hardware detection, path resolution.

Every script imports from here so behaviour (seeds, device, paths) is identical everywhere.
"""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Project root = parent of src/. All relative paths in configs resolve against it,
# so scripts work no matter which directory they are launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


# --------------------------------------------------------------------------- config
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base` (override wins)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def paths_config_file() -> Path:
    """Choose the paths YAML for the current machine.

    Priority: $SKIN_PATHS_CONFIG (explicit override) > Kaggle (if /kaggle/input exists) > local.
    This is how the same code runs on a laptop, in CI and on Kaggle without editing any path.
    """
    if os.environ.get("SKIN_PATHS_CONFIG"):
        return resolve_path(os.environ["SKIN_PATHS_CONFIG"])
    if Path("/kaggle/input").exists():
        return CONFIG_DIR / "paths_kaggle.yaml"
    return CONFIG_DIR / "paths.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/base.yaml + the machine's paths YAML, then overlay an experiment YAML.

    Layering keeps experiment files short: they only list what differs from the defaults.
    """
    with open(CONFIG_DIR / "base.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    with open(paths_config_file(), encoding="utf-8") as f:
        cfg["paths"] = yaml.safe_load(f) or {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Turn a config path (possibly relative) into an absolute Path under the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------- seeding
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy, torch (CPU + all GPUs).

    deterministic=True also forces deterministic cuDNN kernels: exactly reproducible but slower,
    so we only enable it where bit-exactness matters (e.g. tests).
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker a distinct but reproducible numpy/python seed."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- hardware
def get_device(pref: str = "auto"):
    """Return torch.device; 'auto' picks CUDA when available, otherwise CPU."""
    import torch

    if pref == "cpu" or (pref == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if pref in ("cuda", "auto"):
        return torch.device("cuda")
    raise ValueError(f"Unknown device preference: {pref}")


def hardware_summary() -> dict[str, Any]:
    """Describe the compute available (used in logs and to pick a batch size)."""
    import torch

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu_name=props.name,
            vram_gb=round(props.total_memory / 1024**3, 2),
            cuda_version=torch.version.cuda,
        )
    return info


def auto_batch_size(cfg: dict) -> int:
    """Pick a batch size from cfg['hardware']['batch_size'] using detected VRAM.

    4 GB cards cannot hold large batches of 224-300 px images for B3/ConvNeXt, so we use a
    'gpu_small' tier; AMP roughly halves activation memory on top of that.
    """
    import torch

    bs = cfg["hardware"]["batch_size"]
    if not isinstance(bs, dict):  # an explicit integer overrides auto-selection
        return int(bs)
    if not torch.cuda.is_available():
        return int(bs["cpu"])
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return int(bs["gpu_small"] if vram_gb <= 6 else bs["gpu_large"])


def auto_num_workers(cfg: dict) -> int:
    nw = cfg["hardware"].get("num_workers", "auto")
    return min(4, os.cpu_count() or 1) if nw == "auto" else int(nw)


# --------------------------------------------------------------------------- tracking
def results_dir_for(cfg: dict, experiment_name: str | None = None) -> Path:
    """results/<experiment_name>/ — plain files that survive without MLflow (rule 10)."""
    d = resolve_path(cfg["paths"]["results_dir"]) / (experiment_name or cfg["experiment_name"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def start_mlflow_run(cfg: dict, experiment_name: str, run_name: str | None = None):
    """Start an MLflow run stored locally under the `mlflow_uri` folder from the paths YAML.

    Backend: a local SQLite file (<mlflow_uri>/mlflow.db) with artifacts in <mlflow_uri>/artifacts.
    MLflow >= 3.x refuses the old plain-folder file store, and SQLite is its recommended local
    replacement: still a single local file, no server. Browse with:
        mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
    Returns the active run; use as a context manager: `with start_mlflow_run(...):`.
    """
    import mlflow

    root = resolve_path(cfg["paths"]["mlflow_uri"])
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        # Explicit artifact location; otherwise MLflow writes ./mlartifacts in the current dir.
        mlflow.create_experiment(experiment_name,
                                 artifact_location=(root / "artifacts" / experiment_name).as_uri())
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    """{'a': {'b': 1}} -> {'a.b': 1}; MLflow params must be flat."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key + "."))
        else:
            out[key] = v
    return out

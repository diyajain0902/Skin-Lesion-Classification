"""Phase 1 tests: config layering, seeding reproducibility, device detection."""
import random

import numpy as np
import torch

from src.utils import (
    _deep_merge,
    auto_batch_size,
    get_device,
    load_config,
    paths_config_file,
    resolve_path,
    set_seed,
)


def test_config_loads_with_fixed_split_seed():
    cfg = load_config()
    assert cfg["seeds"]["split_seed"] == 2024, "split seed must never change (rule 2)"
    assert cfg["seeds"]["train_seeds"] == [42, 43, 44]
    assert "results_dir" in cfg["paths"]


def test_deep_merge_overrides_nested_keys_only():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    out = _deep_merge(base, {"b": {"y": 3}})
    assert out == {"a": 1, "b": {"x": 1, "y": 3}}
    assert base["b"]["y"] == 2  # original untouched


def test_set_seed_is_reproducible():
    def draw():
        return random.random(), np.random.rand(), torch.rand(1).item()

    set_seed(42)
    first = draw()
    set_seed(42)
    assert draw() == first


def test_device_and_batch_size():
    assert get_device("cpu").type == "cpu"
    assert get_device("auto").type in {"cpu", "cuda"}
    assert auto_batch_size(load_config()) > 0


def test_resolve_path_is_under_project_root():
    assert resolve_path("results").is_absolute()


def test_paths_config_env_override(monkeypatch):
    # The env var must win over auto-detection so any machine can point at its own paths file.
    monkeypatch.setenv("SKIN_PATHS_CONFIG", "configs/paths_kaggle.yaml")
    assert paths_config_file().name == "paths_kaggle.yaml"
    assert load_config()["paths"]["ham10000_dir"].startswith("/kaggle/input")


def test_mlflow_run_logs_to_local_store(tmp_path):
    """Creates a real MLflow run (catches MLflow backend breakages before a Kaggle run does)."""
    import mlflow

    from src.utils import start_mlflow_run

    cfg = load_config()
    cfg["paths"]["mlflow_uri"] = str(tmp_path / "mlruns")
    artifact = tmp_path / "note.txt"
    artifact.write_text("hello")
    with start_mlflow_run(cfg, "pytest_experiment", run_name="t") as run:
        mlflow.log_param("p", 1)
        mlflow.log_metric("m", 0.5)
        mlflow.log_artifact(str(artifact))
    stored = mlflow.get_run(run.info.run_id)
    assert stored.data.params["p"] == "1" and stored.data.metrics["m"] == 0.5
    assert (tmp_path / "mlruns" / "mlflow.db").exists()
    assert list((tmp_path / "mlruns" / "artifacts").rglob("note.txt"))

"""Phase 1 verification: every dependency imports, and the GPU/CPU is detected.

Usage:  python -m src.check_env
"""
from __future__ import annotations

import importlib
import sys

# (import name, pip name) — import names differ from pip names for several packages.
REQUIRED = [
    ("torch", "torch"), ("torchvision", "torchvision"), ("timm", "timm"),
    ("open_clip", "open_clip_torch"), ("segmentation_models_pytorch", "segmentation-models-pytorch"),
    ("albumentations", "albumentations"), ("cv2", "opencv-python-headless"),
    ("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"), ("scipy", "scipy"),
    ("statsmodels", "statsmodels"), ("mapie", "mapie"), ("optuna", "optuna"),
    ("pytorch_grad_cam", "grad-cam"), ("imagehash", "imagehash"), ("PIL", "Pillow"),
    ("matplotlib", "matplotlib"), ("seaborn", "seaborn"), ("mlflow", "mlflow"),
    ("yaml", "PyYAML"), ("kaggle", "kaggle"), ("tqdm", "tqdm"), ("streamlit", "streamlit"),
    ("reportlab", "reportlab"),
]


def main() -> int:
    failed = []
    for mod, pip_name in REQUIRED:
        try:
            m = importlib.import_module(mod)
            print(f"  ok   {mod:<28} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001 — report every failure, not just the first
            # kaggle raises at import if no credentials are configured; that is not an install issue.
            msg = str(e).lower()
            if mod == "kaggle" and ("kaggle.json" in msg or "credentials" in msg):
                print(f"  warn {mod:<28} installed, but no credentials found ({e.__class__.__name__})")
                continue
            failed.append(pip_name)
            print(f"  FAIL {mod:<28} {e.__class__.__name__}: {e}")

    from src.utils import auto_batch_size, get_device, hardware_summary, load_config

    cfg = load_config()
    print("\nHardware:", hardware_summary())
    print("Device   :", get_device(cfg["hardware"]["device"]))
    print("Auto batch size:", auto_batch_size(cfg))

    if failed:
        print("\nMissing packages:", ", ".join(failed))
        return 1
    print("\nAll imports OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

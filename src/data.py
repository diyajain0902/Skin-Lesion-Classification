"""HAM10000 data pipeline.

Responsibilities
  1. Locate (or download) HAM10000 and merge its two image folders into one image_id -> path index.
  2. Lesion-level, class-stratified split: train / val_a / val_b / test (rule 1), saved once to
     splits/splits.csv and reused by every experiment forever (rule 2).
  3. Leakage checks: zero lesion_id overlap between all splits + perceptual-hash near duplicates.
  4. Metadata encoding (age, sex, localization) fitted on TRAIN ONLY (rule 5).
  5. Optional colour constancy / hair removal, albumentations pipelines, PyTorch Dataset.
  6. Class weights, plots and a JSON summary in results/phase2_data/.

CLI:  python -m src.data [--config configs/x.yaml] [--debug] [--download]
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.utils import (
    auto_batch_size,
    auto_num_workers,
    flatten_dict,
    load_config,
    resolve_path,
    results_dir_for,
    seed_worker,
    set_seed,
    start_mlflow_run,
)

SPLITS = ("train", "val_a", "val_b", "test")
CLASS_NAMES = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_IMAGES, EXPECTED_LESIONS = 10015, 7470  # published HAM10000 counts, used as a sanity check


# =========================================================================== 1. locate + load
def find_ham10000_dir(cfg: dict) -> Path:
    """Return the folder holding HAM10000_metadata.csv.

    Kaggle has changed how attached datasets are nested over time, so if the configured path is
    wrong we search /kaggle/input and data_root instead of failing.
    """
    configured = resolve_path(cfg["paths"]["ham10000_dir"])
    if (configured / "HAM10000_metadata.csv").exists():
        return configured
    for root in (Path("/kaggle/input"), resolve_path(cfg["paths"]["data_root"])):
        if root.exists():
            hits = sorted(root.rglob("HAM10000_metadata.csv"))
            if hits:
                return hits[0].parent
    raise FileNotFoundError(
        f"HAM10000_metadata.csv not found under {configured}. On Kaggle, attach dataset "
        f"'{cfg['data']['kaggle_dataset']}'; elsewhere run `python -m src.data --download`."
    )


def download_ham10000(cfg: dict) -> None:
    """Download + unzip via the Kaggle API (needs ~/.kaggle/kaggle.json). Not needed on Kaggle."""
    import kaggle  # imported lazily: it authenticates on import

    out = resolve_path(cfg["paths"]["ham10000_dir"])
    out.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(cfg["data"]["kaggle_dataset"], path=str(out), unzip=True)


def load_metadata(ham_dir: Path) -> pd.DataFrame:
    """Load metadata and attach an absolute image path to every row.

    HAM10000 ships its images in two folders (part_1, part_2); the Kaggle copy also contains a
    lowercase duplicate of each. We 'merge' them by indexing every .jpg by its image_id, so no
    files are copied.
    """
    df = pd.read_csv(ham_dir / "HAM10000_metadata.csv")
    index: dict[str, Path] = {}
    for p in sorted(ham_dir.rglob("*.jpg")):
        index.setdefault(p.stem, p)  # first copy wins; duplicates are byte-identical copies
    df["path"] = df["image_id"].map(lambda i: str(index[i]) if i in index else None)
    n_missing = int(df["path"].isna().sum())
    if n_missing:
        raise FileNotFoundError(f"{n_missing} images listed in metadata were not found on disk")
    # Missing categorical values become an explicit 'unknown' category (HAM uses it already).
    df["sex"] = df["sex"].fillna("unknown")
    df["localization"] = df["localization"].fillna("unknown")
    return df


def sanity_check_metadata(df: pd.DataFrame, classes: list[str]) -> dict:
    """Checks that the data is what we think it is. Returns facts for the summary JSON."""
    assert set(df["dx"].unique()) == set(classes), f"unexpected classes: {sorted(df['dx'].unique())}"
    assert df["image_id"].is_unique, "duplicate image_id rows in metadata"
    # A lesion must have a single diagnosis, otherwise lesion-level stratification is ill-defined.
    dx_per_lesion = df.groupby("lesion_id")["dx"].nunique()
    assert (dx_per_lesion == 1).all(), f"{(dx_per_lesion > 1).sum()} lesions have >1 diagnosis"
    return {
        "n_images": len(df),
        "n_lesions": int(df["lesion_id"].nunique()),
        "matches_published_counts": len(df) == EXPECTED_IMAGES
        and df["lesion_id"].nunique() == EXPECTED_LESIONS,
        "images_per_lesion_max": int(df.groupby("lesion_id").size().max()),
        "age_missing": int(df["age"].isna().sum()),
        "sex_unknown": int((df["sex"] == "unknown").sum()),
        "localization_unknown": int((df["localization"] == "unknown").sum()),
    }


# =========================================================================== 2. splitting
def make_lesion_splits(df: pd.DataFrame, seed: int, split_cfg: dict) -> pd.DataFrame:
    """Assign every image to train/val_a/val_b/test so that all images of a lesion share a split.

    Why by lesion: HAM10000 has several photos of the same lesion. Splitting by image would put
    near-identical photos in train and test, inflating test scores (data leakage).
    Stratification is done on the lesion table (one row per lesion, its diagnosis), so class
    proportions are preserved at lesion level.
    """
    fr = split_cfg
    assert abs(fr["train"] + fr["val"] + fr["test"] - 1.0) < 1e-6, "split fractions must sum to 1"
    # One row per lesion, sorted so the result does not depend on CSV row order.
    lesions = (
        df.groupby("lesion_id")["dx"].first().reset_index().sort_values("lesion_id").reset_index(drop=True)
    )
    train_l, rest_l = train_test_split(
        lesions, train_size=fr["train"], stratify=lesions["dx"], random_state=seed
    )
    val_l, test_l = train_test_split(
        rest_l, test_size=fr["test"] / (fr["val"] + fr["test"]), stratify=rest_l["dx"], random_state=seed
    )
    val_a_l, val_b_l = train_test_split(
        val_l, train_size=fr["val_a_fraction"], stratify=val_l["dx"], random_state=seed
    )
    lesion_to_split = {}
    for name, part in zip(SPLITS, (train_l, val_a_l, val_b_l, test_l)):
        lesion_to_split.update(dict.fromkeys(part["lesion_id"], name))
    out = df[["image_id", "lesion_id", "dx"]].copy()
    out["split"] = out["lesion_id"].map(lesion_to_split)
    return out.sort_values("image_id").reset_index(drop=True)


def assert_no_lesion_overlap(splits: pd.DataFrame) -> dict:
    """Hard leakage check (rule 1). Raises AssertionError on any violation."""
    assert splits["split"].notna().all(), "some images have no split"
    assert set(splits["split"].unique()) <= set(SPLITS), "unknown split name"
    assert splits["image_id"].is_unique, "an image appears in more than one row"
    lesion_sets = {s: set(splits.loc[splits["split"] == s, "lesion_id"]) for s in SPLITS}
    report = {}
    for a, b in combinations(SPLITS, 2):
        shared = lesion_sets[a] & lesion_sets[b]
        report[f"{a}&{b}"] = len(shared)
        assert not shared, f"lesion leakage between {a} and {b}: {sorted(shared)[:5]}"
    return report


def get_or_create_splits(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Load the fixed split file, or create it on the very first run.

    The CSV is committed to git, so splits never change even if a library update changes how
    the random split is computed.
    """
    split_dir = resolve_path(cfg["paths"]["splits_dir"])
    path = split_dir / "splits.csv"
    seed = cfg["seeds"]["split_seed"]
    if path.exists():
        splits = pd.read_csv(path)
        assert set(splits["image_id"]) == set(df["image_id"]), "splits.csv does not match metadata"
        # Check the saved file is still reproducible from the seed; if not, the saved file wins.
        regenerated = make_lesion_splits(df, seed, cfg["data"]["split"])
        same = splits.sort_values("image_id")["split"].tolist() == regenerated["split"].tolist()
        if not same:
            print("WARNING: regenerated split differs from saved splits.csv (library change?). "
                  "Using the saved file, which is the fixed reference.")
        return splits
    splits = make_lesion_splits(df, seed, cfg["data"]["split"])
    split_dir.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False)
    with open(split_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump({"split_seed": seed, "fractions": cfg["data"]["split"],
                   "note": "Fixed forever. Do not regenerate."}, f, indent=2)
    print(f"Created {path}")
    return splits


def split_counts(splits: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Images per class per split, plus lesion counts."""
    tab = pd.crosstab(splits["dx"], splits["split"]).reindex(index=classes, columns=list(SPLITS), fill_value=0)
    tab.loc["total_images"] = tab.sum()
    tab.loc["total_lesions"] = [splits.loc[splits["split"] == s, "lesion_id"].nunique() for s in SPLITS]
    return tab


# =========================================================================== 3. metadata encoding
class MetadataEncoder:
    """age -> standardised (+ missing flag); sex, localization -> one-hot.

    Fit on TRAIN ONLY (rule 5): the age median used for imputation and the mean/std used for
    scaling are train statistics, then applied unchanged to val/test and to app inputs.
    An explicit 'unknown' category is always present, so the app can pass missing sex/location.
    """

    def fit(self, df: pd.DataFrame) -> MetadataEncoder:
        self.age_median = float(df["age"].median())
        age = df["age"].fillna(self.age_median)
        self.age_mean = float(age.mean())
        self.age_std = float(age.std(ddof=0)) or 1.0
        self.sex_categories = sorted(set(df["sex"]) | {"unknown"})
        self.loc_categories = sorted(set(df["localization"]) | {"unknown"})
        return self

    @property
    def feature_names(self) -> list[str]:
        return (["age_scaled", "age_missing"]
                + [f"sex_{c}" for c in self.sex_categories]
                + [f"loc_{c}" for c in self.loc_categories])

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        age = df["age"].astype(float)
        cols = [
            ((age.fillna(self.age_median) - self.age_mean) / self.age_std).to_numpy(),
            age.isna().to_numpy(dtype=float),  # lets the model learn that age was imputed
        ]
        # Unseen categories encode as all zeros instead of crashing.
        sex = df["sex"].fillna("unknown")
        loc = df["localization"].fillna("unknown")
        cols += [(sex == c).to_numpy(dtype=float) for c in self.sex_categories]
        cols += [(loc == c).to_numpy(dtype=float) for c in self.loc_categories]
        return np.column_stack(cols).astype(np.float32)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("age_median", "age_mean", "age_std", "sex_categories", "loc_categories")}

    @classmethod
    def from_dict(cls, d: dict) -> MetadataEncoder:
        enc = cls()
        for k, v in d.items():
            setattr(enc, k, v)
        return enc


def compute_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights w_c = N / (K * n_c), from TRAIN labels.

    A class with average frequency gets weight 1; rare classes (df, vasc) get >1, nv gets <1.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    return counts.sum() / (n_classes * np.maximum(counts, 1.0))


# =========================================================================== 4. preprocessing
def shades_of_gray(img: np.ndarray, p: int = 6) -> np.ndarray:
    """Shades of Gray colour constancy (Finlayson & Trezzi, 2004).

    Estimates the illuminant per channel with a Minkowski p-norm mean and rescales channels so
    the illuminant becomes neutral grey. Reduces colour differences between dermatoscopes.
    p=1 is Grey World, p=inf is Max-RGB; p=6 is the common choice in ISIC challenge work.
    """
    x = img.astype(np.float32)
    illum = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1.0 / p)
    illum = illum / (np.linalg.norm(illum) + 1e-6)
    x = x / (illum * math.sqrt(3) + 1e-6)
    return np.clip(x, 0, 255).astype(np.uint8)


def remove_hair(img: np.ndarray) -> np.ndarray:
    """DullRazor-style hair removal (Lee et al., 1997), simplified.

    1) black-hat morphology on greyscale highlights thin dark structures (hairs);
    2) threshold to a hair mask;  3) inpaint the masked pixels from their surroundings.
    The kernel scales with image size so it behaves the same at 224 or 600 px.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    k = max(3, int(round(min(img.shape[:2]) / 35)) | 1)  # odd kernel, ~17 px at 600 px
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def load_image(path: str, size: int, pre_cfg: dict) -> np.ndarray:
    """Read RGB, resize to size x size, then apply optional deterministic preprocessing.

    Resizing first keeps hair removal cheap (it runs on every load). HAM images are 600x450;
    squashing to a square slightly changes aspect ratio but keeps the whole lesion in view,
    whereas a centre crop could cut off lesion borders.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"could not read image {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if pre_cfg.get("color_constancy"):
        img = shades_of_gray(img, pre_cfg.get("shades_of_gray_p", 6))
    if pre_cfg.get("hair_removal"):
        img = remove_hair(img)
    return img


# =========================================================================== 5. augmentation
def build_transforms(cfg: dict, train: bool, norm: tuple | None = None) -> A.Compose:
    """Albumentations pipelines.

    - eval: resize + normalise only (deterministic).
    - light: flips + 90-degree rotations + small rotation. Dermoscopy has no 'up', so these are
      label-preserving.
    - heavy: light + shift/scale/rotate, colour jitter (simulates different devices/lighting) and
      coarse dropout (forces the model to use more of the lesion, not one patch).
    Reflect padding (not black) is used for affine borders so augmentation does not create black
    corners, which a model could learn as a shortcut.
    """
    size = cfg["data"]["image_size"]
    mean, std = norm or (IMAGENET_MEAN, IMAGENET_STD)
    tail = [A.Normalize(mean=mean, std=std), ToTensorV2()]
    if not train:
        return A.Compose([A.Resize(size, size), *tail])

    base = [A.Resize(size, size), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
    strength = cfg["data"]["augmentation"]
    if strength == "light":
        aug = [A.Affine(rotate=(-20, 20), border_mode=cv2.BORDER_REFLECT_101, p=0.5)]
    elif strength == "heavy":
        aug = [
            A.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1), rotate=(-45, 45),
                     border_mode=cv2.BORDER_REFLECT_101, p=0.7),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.7),
            A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(0.04, 0.12),
                            hole_width_range=(0.04, 0.12), fill=0, p=0.5),
        ]
    else:
        raise ValueError(f"augmentation must be 'light' or 'heavy', got {strength!r}")
    return A.Compose([*base, *aug, *tail])


def compute_train_norm_stats(train_df: pd.DataFrame, cfg: dict, seed: int) -> tuple[tuple, tuple]:
    """Per-channel mean/std estimated on a sample of TRAIN images only (rule 5)."""
    n = min(cfg["data"]["norm_stats_images"], len(train_df))
    sample = train_df.sample(n=n, random_state=seed)
    s = s2 = 0.0
    count = 0
    for path in sample["path"]:
        x = load_image(path, cfg["data"]["image_size"], cfg["data"]["preprocessing"]).astype(np.float64) / 255
        s = s + x.sum(axis=(0, 1))
        s2 = s2 + (x ** 2).sum(axis=(0, 1))
        count += x.shape[0] * x.shape[1]
    mean = s / count
    std = np.sqrt(s2 / count - mean ** 2)
    return tuple(mean.round(4).tolist()), tuple(std.round(4).tolist())


# =========================================================================== 6. dataset
class HAM10000Dataset(Dataset):
    """Returns (image_tensor[3,H,W], metadata_vector[F], label int, lesion_id str).

    lesion_id is returned so evaluation can bootstrap by lesion (rule 9).
    """

    def __init__(self, df: pd.DataFrame, meta: np.ndarray, classes: list[str], transform, cfg: dict):
        assert len(df) == len(meta)
        self.paths = df["path"].tolist()
        self.lesion_ids = df["lesion_id"].tolist()
        self.labels = df["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        self.meta = meta
        self.transform = transform
        self.size = cfg["data"]["image_size"]
        self.pre = cfg["data"]["preprocessing"]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = load_image(self.paths[i], self.size, self.pre)
        x = self.transform(image=img)["image"]
        return x, torch.from_numpy(self.meta[i]), int(self.labels[i]), self.lesion_ids[i]


def debug_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Small subset with every class present (for --debug smoke runs)."""
    per_class = max(1, math.ceil(n / df["dx"].nunique()))
    return df.sample(frac=1.0, random_state=seed).groupby("dx").head(per_class).reset_index(drop=True)


def prepare_data(cfg: dict, debug: bool = False) -> dict:
    """Everything a training script needs: per-split DataFrames, datasets, encoder, weights.

    This is the single entry point used by train/evaluate/etc., so every phase gets the same
    splits, the same train-only encoder and the same normalisation.
    """
    classes = cfg["data"]["classes"]
    df = load_metadata(find_ham10000_dir(cfg))
    splits = get_or_create_splits(cfg, df)
    assert_no_lesion_overlap(splits)
    df = df.merge(splits[["image_id", "split"]], on="image_id", how="left")
    frames = {s: df[df["split"] == s].reset_index(drop=True) for s in SPLITS}
    if debug:
        n = cfg["debug"]["max_samples"]
        frames = {s: debug_subset(f, n if s == "train" else n // 2, cfg["seeds"]["split_seed"])
                  for s, f in frames.items()}

    encoder = MetadataEncoder().fit(frames["train"])  # TRAIN ONLY
    norm = None
    if cfg["data"]["normalization"] == "train":
        norm = compute_train_norm_stats(frames["train"], cfg, cfg["seeds"]["split_seed"])
    datasets = {
        s: HAM10000Dataset(f, encoder.transform(f), classes,
                           build_transforms(cfg, train=(s == "train"), norm=norm), cfg)
        for s, f in frames.items()
    }
    weights = compute_class_weights(datasets["train"].labels, len(classes))
    return {"frames": frames, "datasets": datasets, "encoder": encoder,
            "class_weights": weights, "norm": norm or (IMAGENET_MEAN, IMAGENET_STD)}


def build_dataloaders(cfg: dict, datasets: dict, seed: int, batch_size: int | None = None) -> dict:
    """DataLoaders with seeded shuffling; optional class-balanced sampling for train."""
    bs = batch_size or auto_batch_size(cfg)
    nw = auto_num_workers(cfg)
    g = torch.Generator().manual_seed(seed)
    common = dict(num_workers=nw, worker_init_fn=seed_worker, pin_memory=torch.cuda.is_available(),
                  persistent_workers=nw > 0)
    train_ds = datasets["train"]
    if cfg["data"]["balanced_sampler"]:
        # Each sample drawn with probability ~ 1/(its class count) => classes seen equally often.
        counts = np.bincount(train_ds.labels, minlength=len(cfg["data"]["classes"]))
        w = 1.0 / counts[train_ds.labels]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w),
                                        replacement=True, generator=g)
        train_kw = dict(sampler=sampler)
    else:
        train_kw = dict(shuffle=True, generator=g)
    loaders = {"train": DataLoader(train_ds, batch_size=bs, drop_last=len(train_ds) > bs,
                                   **train_kw, **common)}
    for s in ("val_a", "val_b", "test"):
        loaders[s] = DataLoader(datasets[s], batch_size=bs, shuffle=False, **common)
    return loaders


# =========================================================================== 7. near duplicates
def _phash_one(args: tuple[str, int]) -> int:
    import imagehash
    from PIL import Image

    path, hash_size = args
    with Image.open(path) as im:
        return int(str(imagehash.phash(im, hash_size=hash_size)), 16)


def compute_phashes(paths: list[str], hash_size: int, workers: int) -> np.ndarray:
    """64-bit perceptual hashes (DCT of a downscaled greyscale image). Robust to resizing/JPEG."""
    assert hash_size <= 8, "hash must fit in uint64"
    jobs = [(p, hash_size) for p in paths]
    if workers > 1:
        with Pool(workers) as pool:
            hashes = pool.map(_phash_one, jobs, chunksize=64)
    else:
        hashes = [_phash_one(j) for j in jobs]
    return np.array(hashes, dtype=np.uint64)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Number of set bits per uint64 (= Hamming distance when x = a XOR b)."""
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x).astype(np.int16)
    bits = np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1)
    return bits.sum(axis=-1).astype(np.int16)


def near_duplicate_pairs(hashes: np.ndarray, split: np.ndarray, lesion: np.ndarray,
                         max_distance: int, thresholds: list[int], chunk: int = 512):
    """All-pairs Hamming distances in chunks (n^2/2 comparisons, vectorised).

    Returns (pairs with distance <= max_distance, counts per threshold), where counts separate
    same-lesion pairs (expected: HAM has several photos per lesion) from different-lesion pairs,
    and cross-split pairs (the dangerous kind: possible leakage the lesion_id split missed).
    """
    n = len(hashes)
    pairs = []
    counts = {t: {"same_lesion": 0, "diff_lesion_same_split": 0, "cross_split": 0} for t in thresholds}
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        d = popcount64(hashes[i0:i1, None] ^ hashes[None, :])
        ii, jj = np.nonzero(d <= max(thresholds + [max_distance]))
        ii = ii + i0
        keep = jj > ii  # each unordered pair once, no self-pairs
        ii, jj = ii[keep], jj[keep]
        dist = d[ii - i0, jj]
        same_lesion = lesion[ii] == lesion[jj]
        cross = split[ii] != split[jj]
        for t in thresholds:
            m = dist <= t
            counts[t]["same_lesion"] += int((m & same_lesion).sum())
            counts[t]["diff_lesion_same_split"] += int((m & ~same_lesion & ~cross).sum())
            counts[t]["cross_split"] += int((m & cross).sum())
        m = dist <= max_distance
        pairs.append(pd.DataFrame({"i": ii[m], "j": jj[m], "distance": dist[m]}))
    return pd.concat(pairs, ignore_index=True), counts


def run_duplicate_check(df: pd.DataFrame, cfg: dict, out_dir: Path, debug: bool) -> dict:
    """Hash every image, report near-duplicate pairs across splits, save CSV + example figure."""
    pc = cfg["data"]["phash"]
    sub = df.sort_values("image_id").reset_index(drop=True)
    if debug:
        sub = sub.sample(n=min(cfg["debug"]["phash_images"], len(sub)),
                         random_state=cfg["seeds"]["split_seed"]).reset_index(drop=True)
    cache = resolve_path(cfg["paths"]["data_root"]) / f"phash{pc['hash_size']}.csv"
    cached = pd.read_csv(cache, dtype={"hash": str}) if cache.exists() else pd.DataFrame(columns=["image_id", "hash"])
    known = dict(zip(cached["image_id"], cached["hash"]))
    todo = sub.loc[~sub["image_id"].isin(known), ["image_id", "path"]]
    if len(todo):
        new = compute_phashes(todo["path"].tolist(), pc["hash_size"], auto_num_workers(cfg))
        known.update({i: format(int(h), "016x") for i, h in zip(todo["image_id"], new)})
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"image_id": list(known), "hash": list(known.values())}).to_csv(cache, index=False)
    hashes = np.array([int(known[i], 16) for i in sub["image_id"]], dtype=np.uint64)

    pairs, counts = near_duplicate_pairs(hashes, sub["split"].to_numpy(), sub["lesion_id"].to_numpy(),
                                         pc["max_distance"], pc["report_thresholds"])
    for side, idx in (("a", "i"), ("b", "j")):
        for col in ("image_id", "lesion_id", "dx", "split"):
            pairs[f"{col}_{side}"] = sub[col].to_numpy()[pairs[idx]]
    pairs = pairs.drop(columns=["i", "j"])
    pairs["same_lesion"] = pairs["lesion_id_a"] == pairs["lesion_id_b"]
    pairs["cross_split"] = pairs["split_a"] != pairs["split_b"]
    pairs.sort_values(["cross_split", "distance"], ascending=[False, True]).to_csv(
        out_dir / "near_duplicates.csv", index=False)
    cross = pairs[pairs["cross_split"]]
    plot_duplicate_examples(cross, sub, out_dir / "near_duplicates_cross_split.png")
    return {
        "images_hashed": len(sub),
        "hash_bits": pc["hash_size"] ** 2,
        "near_duplicate_threshold": pc["max_distance"],
        "pair_counts_by_threshold": {str(t): c for t, c in counts.items()},
        "cross_split_pairs_at_threshold": len(cross),
        "cross_split_pairs_by_split_pair": cross.groupby(["split_a", "split_b"]).size()
        .rename("n").reset_index().to_dict("records"),
        "cross_split_pairs_same_dx": int((cross["dx_a"] == cross["dx_b"]).sum()),
    }


# =========================================================================== 8. plots
def plot_class_distribution(counts: pd.DataFrame, classes: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    tab = counts.loc[classes]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    tab.plot.bar(ax=axes[0], logy=True, width=0.8)
    axes[0].set(title="Images per class (log scale)", xlabel="class", ylabel="images")
    (tab / tab.sum()).plot.bar(ax=axes[1], width=0.8, legend=False)
    axes[1].set(title="Class proportion within each split", xlabel="class", ylabel="fraction")
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sample_grid(df: pd.DataFrame, classes: list[str], path: Path, seed: int, per_class: int = 6) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.8, len(classes) * 1.6))
    for r, c in enumerate(classes):
        rows = df[df["dx"] == c]
        rows = rows.sample(n=min(per_class, len(rows)), random_state=seed)
        for k in range(per_class):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(rows):
                ax.imshow(cv2.cvtColor(cv2.imread(rows["path"].iloc[k]), cv2.COLOR_BGR2RGB))
        axes[r, 0].set_title(c, loc="left", fontsize=9)
    fig.suptitle("Training-set examples per class")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_preprocessing_examples(df: pd.DataFrame, cfg: dict, path: Path, seed: int, n: int = 4) -> None:
    import matplotlib.pyplot as plt

    rows = df.sample(n=n, random_state=seed)
    size = cfg["data"]["image_size"]
    fig, axes = plt.subplots(n, 3, figsize=(7.5, n * 2.5))
    titles = ("original", "Shades of Gray (p=6)", "hair removal")
    for r, p in enumerate(rows["path"]):
        imgs = (load_image(p, size, {}), load_image(p, size, {"color_constancy": True}),
                load_image(p, size, {"hair_removal": True}))
        for k, im in enumerate(imgs):
            axes[r, k].imshow(im)
            axes[r, k].axis("off")
            if r == 0:
                axes[r, k].set_title(titles[k], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_duplicate_examples(cross: pd.DataFrame, sub: pd.DataFrame, path: Path, n: int = 8) -> None:
    import matplotlib.pyplot as plt

    if cross.empty:
        return
    paths = dict(zip(sub["image_id"], sub["path"]))
    rows = cross.sort_values("distance").head(n)
    fig, axes = plt.subplots(len(rows), 2, figsize=(5, 2.4 * len(rows)), squeeze=False)
    for r, (_, row) in enumerate(rows.iterrows()):
        for k, side in enumerate("ab"):
            ax = axes[r, k]
            ax.imshow(cv2.cvtColor(cv2.imread(paths[row[f"image_id_{side}"]]), cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_title(f"{row[f'split_{side}']} | {row[f'dx_{side}']} | d={row['distance']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =========================================================================== 9. CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="experiment YAML overlaid on configs/base.yaml")
    ap.add_argument("--debug", action="store_true", help="hash/plot a small subset only")
    ap.add_argument("--download", action="store_true", help="download HAM10000 via the Kaggle API")
    args = ap.parse_args()

    cfg = load_config(args.config)
    classes = cfg["data"]["classes"]
    seed = cfg["seeds"]["split_seed"]
    set_seed(seed)
    exp = "phase2_data_debug" if args.debug else "phase2_data"
    out = results_dir_for(cfg, exp)
    if args.download:
        download_ham10000(cfg)

    # ---- load + split (always on the FULL data: the split is fixed and must cover everything)
    ham_dir = find_ham10000_dir(cfg)
    print(f"HAM10000 at {ham_dir}")
    df = load_metadata(ham_dir)
    facts = sanity_check_metadata(df, classes)
    splits = get_or_create_splits(cfg, df)
    overlap = assert_no_lesion_overlap(splits)
    print("Lesion overlap between splits (all must be 0):", overlap)
    df = df.merge(splits[["image_id", "split"]], on="image_id", how="left")
    counts = split_counts(splits, classes)
    counts.to_csv(out / "split_counts.csv")
    print(counts.to_string())

    train_df = df[df["split"] == "train"]
    encoder = MetadataEncoder().fit(train_df)
    weights = compute_class_weights(train_df["dx"].map({c: i for i, c in enumerate(classes)}).to_numpy(),
                                    len(classes))
    with open(out / "metadata_encoder.json", "w", encoding="utf-8") as f:
        json.dump(encoder.to_dict(), f, indent=2)

    # ---- near-duplicate check
    print("Perceptual-hash near-duplicate check ...")
    dup = run_duplicate_check(df, cfg, out, args.debug)
    print(json.dumps(dup, indent=2))

    # ---- plots
    plot_class_distribution(counts, classes, out / "class_distribution.png")
    plot_sample_grid(train_df, classes, out / "sample_grid.png", seed)
    plot_preprocessing_examples(train_df, cfg, out / "preprocessing_examples.png", seed)

    # ---- dataset / dataloader smoke test: one batch per split, shapes checked
    data = prepare_data(cfg, debug=True)
    loaders = build_dataloaders(cfg, data["datasets"], seed=cfg["seeds"]["train_seeds"][0], batch_size=8)
    size = cfg["data"]["image_size"]
    batch_shapes = {}
    for s, loader in loaders.items():
        x, m, y, lid = next(iter(loader))
        assert x.shape[1:] == (3, size, size), x.shape
        assert m.shape[1] == encoder.n_features and m.dtype == torch.float32, m.shape
        assert y.dtype == torch.int64 and len(lid) == len(y)
        assert torch.isfinite(x).all() and torch.isfinite(m).all()
        batch_shapes[s] = {"image": list(x.shape), "metadata": list(m.shape), "label": list(y.shape)}
    print("Batch shapes:", batch_shapes)

    summary = {
        "experiment": exp,
        "metadata_facts": facts,
        "split_seed": seed,
        "lesion_overlap": overlap,
        "lesion_overlap_check": "passed",
        "images_per_split": counts.loc["total_images"].to_dict(),
        "lesions_per_split": counts.loc["total_lesions"].to_dict(),
        "lesion_fraction_per_split": (counts.loc["total_lesions"] / counts.loc["total_lesions"].sum())
        .round(4).to_dict(),
        "class_weights_train": dict(zip(classes, np.round(weights, 4).tolist())),
        "metadata_features": encoder.feature_names,
        "near_duplicates": dup,
        "batch_shapes_debug_loader": batch_shapes,
    }
    with open(out / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=int)

    with start_mlflow_run(cfg, exp, run_name="data_pipeline"):
        import mlflow

        mlflow.log_params({k: v for k, v in flatten_dict(cfg["data"]).items() if not isinstance(v, list)})
        mlflow.log_param("split_seed", seed)
        mlflow.log_metrics({f"images_{k}": float(v) for k, v in summary["images_per_split"].items()})
        mlflow.log_metric("cross_split_near_duplicates", dup["cross_split_pairs_at_threshold"])
        mlflow.log_artifacts(str(out))
    print(f"\nPhase 2 data pipeline OK. Outputs in {out}")


if __name__ == "__main__":
    main()

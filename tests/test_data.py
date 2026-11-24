"""Phase 2 tests. Synthetic data only, so they run in CI without downloading HAM10000."""
import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.data import (
    SPLITS,
    HAM10000Dataset,
    build_from_frames,
    MetadataEncoder,
    assert_no_lesion_overlap,
    build_transforms,
    compute_class_weights,
    make_lesion_splits,
    near_duplicate_pairs,
    popcount64,
    remove_hair,
    shades_of_gray,
)
from src.utils import load_config

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
# Lesions per class, skewed like HAM10000 (nv dominates, df/vasc rare).
LESIONS_PER_CLASS = {"akiec": 60, "bcc": 80, "bkl": 120, "df": 40, "mel": 120, "nv": 600, "vasc": 40}
SPLIT_CFG = {"train": 0.70, "val": 0.15, "test": 0.15, "val_a_fraction": 0.5}


@pytest.fixture(scope="module")
def meta() -> pd.DataFrame:
    """Fake metadata: 1-4 images per lesion, one diagnosis per lesion, some missing values."""
    rng = np.random.default_rng(0)
    rows, img = [], 0
    for dx, n_les in LESIONS_PER_CLASS.items():
        for k in range(n_les):
            lid = f"L_{dx}_{k:04d}"
            age = float(rng.integers(5, 90)) if rng.random() > 0.05 else np.nan
            sex = rng.choice(["male", "female", "unknown"], p=[0.5, 0.45, 0.05])
            loc = rng.choice(["back", "face", "trunk", "unknown"])
            for _ in range(rng.integers(1, 5)):
                rows.append(dict(image_id=f"I_{img:06d}", lesion_id=lid, dx=dx, age=age, sex=sex, localization=loc))
                img += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def splits(meta) -> pd.DataFrame:
    return make_lesion_splits(meta, seed=2024, split_cfg=SPLIT_CFG)


# ------------------------------------------------------------------ split logic
def test_every_image_assigned_once_and_no_lesion_overlap(meta, splits):
    assert len(splits) == len(meta)
    assert set(splits["image_id"]) == set(meta["image_id"])
    report = assert_no_lesion_overlap(splits)
    assert len(report) == 6 and all(v == 0 for v in report.values())  # all 6 split pairs
    assert (splits.groupby("lesion_id")["split"].nunique() == 1).all()


def test_overlap_assertion_catches_leakage(splits):
    # Move ONE image of a multi-image lesion into another split: must be detected.
    multi = splits.groupby("lesion_id").size()
    lid = multi[multi > 1].index[0]
    leaked = splits.copy()
    i = leaked.index[leaked["lesion_id"] == lid][0]
    leaked.loc[i, "split"] = "test" if leaked.loc[i, "split"] != "test" else "train"
    with pytest.raises(AssertionError, match="lesion leakage"):
        assert_no_lesion_overlap(leaked)


def test_lesion_level_proportions(splits):
    les = splits.drop_duplicates("lesion_id")["split"].value_counts(normalize=True)
    assert les["train"] == pytest.approx(0.70, abs=0.01)
    assert les["test"] == pytest.approx(0.15, abs=0.01)
    assert les["val_a"] + les["val_b"] == pytest.approx(0.15, abs=0.01)
    assert les["val_a"] == pytest.approx(les["val_b"], abs=0.01)


def test_stratified_every_class_in_every_split(splits):
    les = splits.drop_duplicates("lesion_id")
    for s in SPLITS:
        part = les[les["split"] == s]
        assert set(part["dx"]) == set(CLASSES), f"{s} is missing a class"
    # Class proportions in test should match train within a few percent (stratification).
    p_train = les[les["split"] == "train"]["dx"].value_counts(normalize=True)
    p_test = les[les["split"] == "test"]["dx"].value_counts(normalize=True)
    assert (p_train - p_test).abs().max() < 0.03


def test_split_is_deterministic_and_seed_dependent(meta, splits):
    again = make_lesion_splits(meta, seed=2024, split_cfg=SPLIT_CFG)
    pd.testing.assert_frame_equal(splits, again)
    # Row order of the input must not change the split.
    shuffled = make_lesion_splits(meta.sample(frac=1, random_state=1), seed=2024, split_cfg=SPLIT_CFG)
    pd.testing.assert_frame_equal(splits, shuffled)
    other = make_lesion_splits(meta, seed=7, split_cfg=SPLIT_CFG)
    assert not splits["split"].equals(other["split"])


# ------------------------------------------------------------------ metadata + weights
def test_metadata_encoder_uses_train_statistics_only(meta, splits):
    m = meta.merge(splits[["image_id", "split"]], on="image_id")
    train, test = m[m["split"] == "train"], m[m["split"] == "test"].copy()
    enc = MetadataEncoder().fit(train)
    assert enc.age_median == pytest.approx(train["age"].median())
    test.loc[test.index[0], "age"] = np.nan
    test.loc[test.index[1], "localization"] = "never_seen_site"
    X = enc.transform(test)
    assert X.shape == (len(test), enc.n_features) and X.dtype == np.float32
    assert np.isfinite(X).all()
    names = enc.feature_names
    # Missing age -> imputed with TRAIN median (scaled) and flagged.
    assert X[0, names.index("age_missing")] == 1.0
    assert X[0, names.index("age_scaled")] == pytest.approx((enc.age_median - enc.age_mean) / enc.age_std)
    # Unseen category -> all-zero one-hot instead of a crash.
    loc_cols = [i for i, n in enumerate(names) if n.startswith("loc_")]
    assert X[1, loc_cols].sum() == 0
    # Round-trip serialisation (needed by the app).
    enc2 = MetadataEncoder.from_dict(enc.to_dict())
    np.testing.assert_array_equal(enc2.transform(test), X)


def test_class_weights_inverse_frequency():
    labels = np.array([0] * 80 + [1] * 15 + [2] * 5)
    w = compute_class_weights(labels, 3)
    assert w[2] > w[1] > w[0]
    assert np.sum(w * np.bincount(labels)) == pytest.approx(len(labels))  # weighted count = N


# ------------------------------------------------------------------ near duplicates
def test_popcount_and_duplicate_pairs():
    x = np.array([0, 0b1011, 2**64 - 1], dtype=np.uint64)
    assert popcount64(x).tolist() == [0, 3, 64]
    # distances: (0,1)=1 cross-split, (0,2)=4 same split, (1,2)=3, last hash >=16 from all
    hashes = np.array([0b0000, 0b0001, 0b1111, 0xFFFF000000000000], dtype=np.uint64)
    split = np.array(["train", "test", "train", "test"])
    lesion = np.array(["a", "b", "c", "d"])
    pairs, counts = near_duplicate_pairs(hashes, split, lesion, max_distance=1, thresholds=[0, 1, 4])
    assert len(pairs) == 1 and pairs.iloc[0][["i", "j", "distance"]].tolist() == [0, 1, 1]
    assert counts[1]["cross_split"] == 1
    assert counts[4]["diff_lesion_same_split"] == 1  # 0b0000 vs 0b1111, both train


# ------------------------------------------------------------------ images
@pytest.fixture()
def rgb_image() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (450, 600, 3), dtype=np.uint8)


def test_preprocessing_keeps_shape_and_dtype(rgb_image):
    for fn in (shades_of_gray, remove_hair):
        out = fn(rgb_image)
        assert out.shape == rgb_image.shape and out.dtype == np.uint8


@pytest.mark.parametrize("aug,train", [("light", True), ("heavy", True), ("light", False)])
def test_transform_output_shape(rgb_image, aug, train):
    cfg = load_config()
    cfg["data"]["augmentation"] = aug
    x = build_transforms(cfg, train=train)(image=rgb_image)["image"]
    size = cfg["data"]["image_size"]
    assert isinstance(x, torch.Tensor) and x.shape == (3, size, size) and x.dtype == torch.float32


def test_dataset_item(tmp_path, rgb_image):
    cfg = load_config()
    paths = []
    for k in range(3):
        p = tmp_path / f"img{k}.jpg"
        cv2.imwrite(str(p), rgb_image)
        paths.append(str(p))
    df = pd.DataFrame({"path": paths, "lesion_id": ["L1", "L1", "L2"], "dx": ["mel", "mel", "nv"],
                       "age": [50.0, np.nan, 30.0], "sex": ["male", "female", "unknown"],
                       "localization": ["back", "face", "back"]})
    enc = MetadataEncoder().fit(df)
    ds = HAM10000Dataset(df, enc.transform(df), CLASSES, build_transforms(cfg, train=False), cfg)
    x, m, y, lid = ds[0]
    size = cfg["data"]["image_size"]
    assert x.shape == (3, size, size)
    assert m.shape == (enc.n_features,) and m.dtype == torch.float32
    assert y == CLASSES.index("mel") and lid == "L1"


def test_debug_subset_keeps_full_metadata_width(meta, splits):
    """Regression: the encoder must be fit on the FULL train split, not the --debug subset.

    First Kaggle run: a rare body site missing from the 128-image debug subset dropped a one-hot
    column, so debug batches had 16 features instead of the full encoder's width.
    """
    cfg = load_config()
    m = meta.merge(splits[["image_id", "split"]], on="image_id").assign(path="unused.jpg")
    train_idx = m.index[m["split"] == "train"]
    m.loc[train_idx[0], "localization"] = "rare_site"  # only one train image has it
    frames = {s: m[m["split"] == s].reset_index(drop=True) for s in SPLITS}
    full = build_from_frames(cfg, frames, debug=False)
    dbg = build_from_frames(cfg, frames, debug=True)
    width = full["encoder"].n_features
    assert "loc_rare_site" in dbg["encoder"].feature_names
    assert dbg["encoder"].n_features == width
    for s in SPLITS:
        assert dbg["datasets"][s].meta.shape[1] == width
    assert len(dbg["datasets"]["train"]) <= cfg["debug"]["max_samples"] + len(CLASSES)
    np.testing.assert_allclose(dbg["class_weights"], full["class_weights"])

# =============================================================================
# TEST: FROZEN SPLIT AND TRAIN-ONLY SCALING  (data/protocol.py)
# =============================================================================

import types
import tempfile
import json
from pathlib import Path

import numpy as np

from src.data.protocol import stratified_indices, finalize_dataset


def test_split_sizes_disjoint_and_reproducible():
    labels = np.r_[np.zeros(250), np.ones(250)]
    np.random.default_rng(0).shuffle(labels)
    train, val, test = stratified_indices(labels, 0.8, 0.1, 123)

    assert (len(train), len(val), len(test)) == (400, 50, 50)
    assert len(set(train) | set(val) | set(test)) == 500
    for part, size in ((train, 200), (val, 25), (test, 25)):
        assert (labels[part] == 0).sum() == size and (labels[part] == 1).sum() == size

    again = stratified_indices(labels, 0.8, 0.1, 123)
    for a, b in zip((train, val, test), again):
        assert np.array_equal(a, b)


def test_scale_uses_training_cases_only():
    n_cases, n_points = 20, 6
    rng = np.random.default_rng(1)
    temperature = rng.uniform(320, 600, size=(n_cases, n_points)).astype(np.float32)
    mask = np.ones((n_cases, n_points), dtype=np.float32)
    meta = np.zeros((n_cases, 5), dtype=np.float32)
    meta[10:, 0] = 1

    with tempfile.TemporaryDirectory() as folder:
        cfg = types.SimpleNamespace(train_fraction=0.8, val_fraction=0.1, split_seed=123,
                                    dataset_path=str(Path(folder) / "dataset.npz"),
                                    Nx=3, Ny=2, Lx=1.0, Ly=1.0, maxh=0.1, order=2, kappa=1.0,
                                    h_hole=1.0, T_inf=300.0, q_top=1.0, dataset_family="combined")
        # MAKE ONE TEST CASE MUCH HOTTER: IT MUST NOT CHANGE THE SCALE
        train, val, test = stratified_indices(meta[:, 0], 0.8, 0.1, 123)
        temperature[test[0]] = 2000.0

        finalize_dataset(cfg, np.zeros_like(temperature), mask, temperature, meta,
                         ["case"] * n_cases, np.zeros((n_points, 2)))
        # "with" closes the file, otherwise Windows cannot delete the folder afterwards
        with np.load(cfg.dataset_path) as data:
            T_max = float(data["T_max"][0])
            hottest_test_value = float(data["temp"][test[0]].max())
        assert abs(T_max - temperature[train].max()) < 1e-3
        assert hottest_test_value > 1.0
        manifest = json.loads((Path(folder) / "dataset_manifest.json").read_text())
        assert manifest["temperature_scale_source"] == "training solid points only"


# =============================================================================
# TEST: HALTON SEQUENCE OF THE DATA-SIZE STUDY  (data/scaling.py)
#   known first values, and even spread of every prefix (nested sets)
# =============================================================================
def test_halton_values_and_prefix_spread():
    from src.data.scaling import radical_inverse, halton_sequence

    assert [radical_inverse(i, 2) for i in (1, 2, 3, 6)] == [0.5, 0.25, 0.75, 0.375]
    assert abs(radical_inverse(1, 3) - 1.0 / 3.0) < 1e-15
    assert abs(radical_inverse(4, 3) - 4.0 / 9.0) < 1e-15

    points = halton_sequence(3200, 4)
    for n in (200, 400, 800, 1600, 3200):
        prefix = points[:n]
        # EVERY ONE OF 10 EQUAL BINS PER DIMENSION HOLDS 10 % OF THE POINTS (+-1.5 %)
        for j in range(4):
            counts = np.histogram(prefix[:, j], bins=10, range=(0.0, 1.0))[0]
            assert np.all(np.abs(counts / n - 0.1) <= 0.015), (n, j, counts)

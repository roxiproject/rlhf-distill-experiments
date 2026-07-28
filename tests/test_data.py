import numpy as np

from rlhf_distill.data import generate_blobs, generate_moons, standardize, train_val_split


def test_generate_blobs_shape_and_classes():
    x, y = generate_blobs(n_samples=300, n_classes=3, seed=0)
    assert x.shape == (300, 2)
    assert y.shape == (300,)
    assert set(np.unique(y)) == {0, 1, 2}


def test_generate_blobs_deterministic_with_seed():
    x1, y1 = generate_blobs(seed=42)
    x2, y2 = generate_blobs(seed=42)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_generate_blobs_different_seeds_differ():
    x1, _ = generate_blobs(seed=1)
    x2, _ = generate_blobs(seed=2)
    assert not np.allclose(x1, x2)


def test_generate_moons_shape():
    x, y = generate_moons(n_samples=400, seed=0)
    assert x.shape == (400, 2)
    assert y.shape == (400,)
    assert set(np.unique(y)) == {0, 1}


def test_generate_moons_balanced_classes():
    x, y = generate_moons(n_samples=500, seed=0)
    counts = np.bincount(y)
    assert abs(int(counts[0]) - int(counts[1])) <= 1


def test_generate_moons_deterministic():
    x1, y1 = generate_moons(seed=7)
    x2, y2 = generate_moons(seed=7)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_generate_moons_noise_zero_still_valid():
    x, y = generate_moons(n_samples=100, noise=0.0, seed=0)
    assert np.all(np.isfinite(x))


def test_train_val_split_sizes():
    x, y = generate_moons(n_samples=200, seed=0)
    x_tr, y_tr, x_val, y_val = train_val_split(x, y, val_frac=0.25, seed=0)
    assert len(y_val) == 50
    assert len(y_tr) == 150
    assert x_tr.shape[1] == 2


def test_train_val_split_no_overlap():
    x, y = generate_moons(n_samples=100, seed=0)
    x_tr, y_tr, x_val, y_val = train_val_split(x, y, val_frac=0.3, seed=1)
    # every val point's row should not literally appear identically in train
    # (values are continuous so exact collision is a strong indicator of overlap)
    for row in x_val:
        assert not np.any(np.all(np.isclose(x_tr, row), axis=1))


def test_standardize_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1000, 3)) * 5 + 10
    x_scaled, mean, std = standardize(x)
    np.testing.assert_allclose(x_scaled.mean(axis=0), np.zeros(3), atol=1e-8)
    np.testing.assert_allclose(x_scaled.std(axis=0), np.ones(3), atol=1e-8)


def test_standardize_reapply_with_given_stats():
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((100, 2)) * 3 + 1
    x_val = rng.standard_normal((20, 2)) * 3 + 1
    _, mean, std = standardize(x_train)
    x_val_scaled, mean2, std2 = standardize(x_val, mean=mean, std=std)
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_array_equal(std, std2)


def test_standardize_handles_zero_variance_column():
    x = np.ones((10, 2))
    x_scaled, mean, std = standardize(x)
    assert np.all(np.isfinite(x_scaled))

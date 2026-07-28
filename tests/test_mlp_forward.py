import numpy as np

from rlhf_distill.mlp import MLP


def test_output_shape_single_hidden_layer():
    m = MLP([3, 5, 2], seed=0)
    x = np.random.default_rng(0).standard_normal((10, 3))
    out = m.forward(x)
    assert out.shape == (10, 2)


def test_output_shape_multi_hidden_layer():
    m = MLP([4, 8, 6, 3], seed=1)
    x = np.random.default_rng(1).standard_normal((7, 4))
    out = m.forward(x)
    assert out.shape == (7, 3)


def test_no_hidden_layer_is_linear():
    m = MLP([3, 2], seed=2)
    x = np.random.default_rng(2).standard_normal((5, 3))
    out = m.forward(x, cache=False)
    expected = x @ m.weights[0] + m.biases[0]
    np.testing.assert_allclose(out, expected)


def test_relu_applied_between_hidden_layers():
    m = MLP([2, 3, 2], seed=3)
    x = np.random.default_rng(3).standard_normal((6, 2))
    m.forward(x)
    hidden_pre = m._cache["pre_activations"][0]
    hidden_post = m._cache["activations"][1]
    np.testing.assert_allclose(hidden_post, np.maximum(hidden_pre, 0.0))


def test_final_layer_has_no_activation():
    m = MLP([2, 3, 2], seed=4)
    x = np.random.default_rng(4).standard_normal((6, 2))
    out = m.forward(x)
    final_pre = m._cache["pre_activations"][-1]
    np.testing.assert_allclose(out, final_pre)


def test_num_params_matches_manual_count():
    m = MLP([3, 5, 2], seed=0)
    expected = (3 * 5 + 5) + (5 * 2 + 2)
    assert m.num_params() == expected


def test_get_set_params_roundtrip():
    m = MLP([3, 4, 2], seed=0)
    params = [p.copy() for p in m.get_params()]
    m.weights[0] += 1.0
    m.set_params(params)
    np.testing.assert_allclose(m.weights[0], params[0])


def test_predict_returns_valid_class_indices():
    m = MLP([2, 4, 3], seed=5)
    x = np.random.default_rng(5).standard_normal((20, 2))
    preds = m.predict(x)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_predict_proba_sums_to_one():
    m = MLP([2, 4, 3], seed=6)
    x = np.random.default_rng(6).standard_normal((15, 2))
    proba = m.predict_proba(x)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(15), atol=1e-10)


def test_forward_deterministic_given_same_weights():
    m = MLP([3, 5, 2], seed=7)
    x = np.random.default_rng(7).standard_normal((4, 3))
    out1 = m.forward(x, cache=False)
    out2 = m.forward(x, cache=False)
    np.testing.assert_array_equal(out1, out2)


def test_invalid_layer_sizes_raises():
    import pytest

    with pytest.raises(ValueError):
        MLP([5], seed=0)


def test_backward_without_forward_raises():
    import pytest

    m = MLP([2, 3, 1], seed=0)
    with pytest.raises(RuntimeError):
        m.backward(np.zeros((1, 1)))


def test_different_seeds_give_different_weights():
    m1 = MLP([3, 4, 2], seed=1)
    m2 = MLP([3, 4, 2], seed=2)
    assert not np.allclose(m1.weights[0], m2.weights[0])

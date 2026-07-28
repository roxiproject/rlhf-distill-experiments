import numpy as np

from rlhf_distill.optim import Adam


def test_adam_reduces_quadratic_loss():
    p = np.array([5.0, -3.0])
    params = [p]
    opt = Adam(params, lr=0.1)
    for _ in range(200):
        grads = [2 * params[0]]  # gradient of sum(x^2)
        opt.step(params, grads)
    assert np.all(np.abs(params[0]) < 0.1)


def test_adam_state_shapes_match_params():
    params = [np.zeros((3, 4)), np.zeros(4)]
    opt = Adam(params, lr=1e-3)
    assert opt.m[0].shape == (3, 4)
    assert opt.v[1].shape == (4,)


def test_adam_step_count_increments():
    params = [np.zeros(2)]
    opt = Adam(params)
    for i in range(5):
        opt.step(params, [np.ones(2)])
    assert opt.t == 5


def test_adam_zero_grad_does_not_change_params():
    params = [np.array([1.0, 2.0])]
    original = params[0].copy()
    opt = Adam(params, lr=0.1)
    opt.step(params, [np.zeros(2)])
    np.testing.assert_allclose(params[0], original)


def test_adam_mismatched_length_raises():
    import pytest

    params = [np.zeros(2), np.zeros(3)]
    opt = Adam(params)
    with pytest.raises(ValueError):
        opt.step([np.zeros(2)], [np.zeros(2)])


def test_adam_moves_params_in_negative_gradient_direction_initially():
    params = [np.array([10.0])]
    opt = Adam(params, lr=0.5)
    opt.step(params, [np.array([1.0])])  # positive gradient
    assert params[0][0] < 10.0


def test_adam_multiple_param_groups_independent():
    p1 = np.array([1.0])
    p2 = np.array([100.0])
    params = [p1, p2]
    opt = Adam(params, lr=0.1)
    opt.step(params, [np.array([1.0]), np.array([0.0])])
    assert params[1][0] == 100.0
    assert params[0][0] < 1.0

import sys
import numpy as np

sys.path.insert(0, ".")
from gradscaler.scaler import GradScaler, Parameter, DummyOptimizer
from gradscaler.verify import verify_unscaled_grad


def test_unscale_returns_true_unscaled_grad():
    p1 = Parameter([1.0, 2.0])
    p1.grad = np.array([1024.0, 2048.0], dtype=np.float64)
    opt = DummyOptimizer([{"params": [p1]}])
    scaler = GradScaler(init_scale=1024.0)

    grads = scaler.unscale_(opt)
    expected = np.array([1.0, 2.0], dtype=np.float64)

    assert len(grads) == 1
    assert np.allclose(grads[0][0], expected)
    assert np.allclose(p1.grad, expected)


def test_verify_unscaled_grad_accuracy():
    p1 = Parameter([0.5, -0.5])
    p1.grad = np.array([256.0, -256.0], dtype=np.float64)
    opt = DummyOptimizer([{"params": [p1]}])
    scaler = GradScaler(init_scale=512.0)

    expected_true = [[np.array([0.5, -0.5], dtype=np.float64)]]
    err = verify_unscaled_grad(scaler, opt, expected_true)
    assert err < 1e-4

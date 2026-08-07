import numpy as np
from loftq.residual import init_loftq_residual
from loftq.dora import dora_forward
from loftq.params import compute_param_delta


def test_loftq_reconstruction():
    W = np.random.randn(16, 32)
    W_q, A, B = init_loftq_residual(W, 4)
    assert W_q.shape == W.shape
    assert A.shape == (4, 32)
    assert B.shape == (16, 4)


def test_dora_forward_differs():
    W_q = np.zeros((16, 32))
    A = np.ones((4, 32)) * 0.1
    B = np.ones((16, 4)) * 0.1
    x = np.ones((2, 32))
    g = np.ones(16) * 1.5
    out_standard = dora_forward(x, W_q, A, B, g, use_dora=False)
    out_dora = dora_forward(x, W_q, A, B, g, use_dora=True)
    assert not np.allclose(out_standard, out_dora)


def test_param_delta_calculation():
    delta_off = compute_param_delta(32, 16, 4, use_dora=False)
    delta_on = compute_param_delta(32, 16, 4, use_dora=True)
    assert delta_off == 0
    assert delta_on == 16

import numpy as np
from distill.metrics import compute_hidden_state_mse, compute_cosine_loss
from distill.mapping import evaluate_layer_mapping


def test_mse_basic():
    s = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    val = compute_hidden_state_mse(s, t)
    assert abs(val) < 1e-5
    assert isinstance(val, float)


def test_cosine_loss_basic():
    s = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    val = compute_cosine_loss(s, t)
    assert abs(val) < 1e-5
    assert isinstance(val, float)


def test_mapping_basic():
    s = np.ones((4, 8), dtype=np.float32)
    t = np.ones((8, 8), dtype=np.float32)
    val = evaluate_layer_mapping(s, t, "uniform")
    assert isinstance(val, float)
    assert val >= 0.0
    val_k = evaluate_layer_mapping(s, t, "every_k")
    assert isinstance(val_k, float)
    assert val_k >= 0.0

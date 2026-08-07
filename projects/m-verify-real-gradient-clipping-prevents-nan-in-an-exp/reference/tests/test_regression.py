import sys
import numpy as np

sys.path.insert(0, ".")
from gradclip.clipping import clip_grad_norm
from gradclip.quant import simulate_nf4_cycles
from gradclip.loop import run_training_step


def test_clip_respects_max_norm():
    grads = [np.array([10.0, 20.0], dtype=np.float32)]
    clipped, norm = clip_grad_norm(grads, 1.0)
    assert norm > 1.0
    new_norm = np.sqrt(sum(np.sum(np.square(g)) for g in clipped))
    assert new_norm <= 1.0 + 1e-5


def test_nan_prevention():
    weights = np.array([1.0, 2.0], dtype=np.float32)
    grad = np.array([1e10, 1e10], dtype=np.float32)
    _, has_nan = run_training_step(weights, grad, 1.0, 0.1)
    assert not has_nan


def test_nf4_error_accumulation():
    tensor = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    _, error = simulate_nf4_cycles(tensor, 5)
    assert error >= 0.0

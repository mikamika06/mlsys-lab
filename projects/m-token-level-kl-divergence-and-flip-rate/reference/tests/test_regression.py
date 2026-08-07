import sys
import numpy as np

sys.path.insert(0, ".")
from eval_metrics.divergence import compute_kl_divergence, compute_flip_rate


def test_self_kl_divergence_is_zero():
    rng = np.random.RandomState(42)
    logits = rng.randn(10, 50, 100)
    kl = compute_kl_divergence(logits, logits)
    assert np.allclose(kl, 0.0, atol=1e-6), f"Self KL divergence non-zero: max={np.max(kl)}"


def test_identical_logits_zero_flip_rate():
    rng = np.random.RandomState(42)
    logits = rng.randn(10, 50, 100)
    flip = compute_flip_rate(logits, logits)
    assert flip == 0.0, f"Identical logits had non-zero flip rate: {flip}"

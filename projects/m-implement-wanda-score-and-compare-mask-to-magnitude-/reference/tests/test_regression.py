import numpy as np
from pruning.sparsegpt import obs_prune
from pruning.wanda import magnitude_mask

def test_obs_updates_weights():
    rng = np.random.RandomState(42)
    W = rng.randn(4, 16)
    X = rng.randn(32, 16)

    W_new, mask = obs_prune(W, X, 0.5)
    W_masked = W * mask

    diff = np.abs(W_new - W_masked).sum()
    assert diff > 1e-4, "OBS did not update the remaining unpruned weights"

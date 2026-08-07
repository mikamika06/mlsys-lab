import sys
import numpy as np

sys.path.insert(0, ".")
from kvcache.ablation import measure_sink_ablation_blowup
from kvcache.mask import reconstruct_kept_mask


def test_sink_ablation_error_positive():
    rng = np.random.default_rng(42)
    k = rng.standard_normal((1, 2, 16, 32))
    v = rng.standard_normal((1, 2, 16, 32))
    q = rng.standard_normal((1, 2, 1, 32))
    err, ratio = measure_sink_ablation_blowup(k, v, q)
    assert err >= 0.0
    assert ratio >= 0.0


def test_mask_reconstruction_shape_and_type():
    dump = {"indices": [0, 2, 5]}
    mask = reconstruct_kept_mask(dump, 8)
    assert mask.shape == (8,)
    assert mask.dtype == bool
    assert mask[0] and not mask[1] and mask[2]

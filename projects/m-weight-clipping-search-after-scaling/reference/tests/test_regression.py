import sys
import numpy as np

sys.path.insert(0, ".")
from awq_clip.quant import search_clipping, quantize_and_reconstruct


def test_clipping_never_worse_than_unclipped():
    np.random.seed(42)
    w = np.random.randn(256, 128)
    w[0, 5] = 15.0
    w[1, 10] = -12.0

    w_reshaped = w.reshape(-1, 128)
    w_max = np.max(np.abs(w_reshaped), axis=1, keepdims=True)

    best_idx, opt_max = search_clipping(w, n_bits=4, group_size=128, n_grid=100)

    w_rec_clipped = quantize_and_reconstruct(w_reshaped, opt_max, n_bits=4)
    err_clipped = np.sum((w_reshaped - w_rec_clipped)**2, axis=1)

    w_rec_unclipped = quantize_and_reconstruct(w_reshaped, w_max, n_bits=4)
    err_unclipped = np.sum((w_reshaped - w_rec_unclipped)**2, axis=1)

    assert np.all(err_clipped <= err_unclipped + 1e-4), "Clipped MSE exceeded unclipped MSE"

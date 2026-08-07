import numpy as np
from calib.nvfp4 import nvfp4_round_trip


def test_nvfp4_block_scale_integrity():
    np.random.seed(42)
    block_a = np.random.uniform(-10.0, 10.0, size=(16,))
    block_b = np.random.uniform(-0.1, 0.1, size=(16,))
    combined = np.concatenate([block_a, block_b])

    reconstructed = nvfp4_round_trip(combined, block_size=16)

    err_a = np.max(np.abs(combined[:16] - reconstructed[:16]))
    err_b = np.max(np.abs(combined[16:] - reconstructed[16:]))

    assert err_a < 2.0, f"Block A scale corrupted: max err {err_a}"
    assert err_b < 0.05, f"Block B scale corrupted: max err {err_b}"

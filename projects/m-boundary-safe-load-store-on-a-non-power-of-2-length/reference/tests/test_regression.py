import numpy as np
from triton_bounds.ops import catch_unmasked_store


def test_unmasked_store_detection():
    x = np.ones(100, dtype=np.float32)
    caught, msg = catch_unmasked_store(x, 100, block_size=64)
    assert caught is True

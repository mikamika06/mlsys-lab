import numpy as np
from prefixcache.dynamic import DynamicCache


def test_dynamic_cache_crop_all_layers():
    cache = DynamicCache()
    k1 = np.ones((1, 4, 20, 16), dtype=np.float32)
    v1 = np.ones((1, 4, 20, 16), dtype=np.float32)
    k2 = np.ones((1, 4, 20, 16), dtype=np.float32) * 2
    v2 = np.ones((1, 4, 20, 16), dtype=np.float32) * 2

    cache.update(k1, v1, 0)
    cache.update(k2, v2, 1)

    assert cache.get_seq_length(0) == 20
    assert cache.get_seq_length(1) == 20

    cache.crop(10)

    assert cache.get_seq_length(0) == 10
    assert cache.get_seq_length(1) == 10
    assert cache.key_cache[0].shape[2] == 10
    assert cache.key_cache[1].shape[2] == 10


def test_dynamic_cache_resume_after_crop():
    cache = DynamicCache()
    k = np.ones((1, 2, 15, 8), dtype=np.float32)
    v = np.ones((1, 2, 15, 8), dtype=np.float32)
    cache.update(k, v, 0)

    cache.crop(8)
    assert cache.get_seq_length(0) == 8

    k_new = np.ones((1, 2, 1, 8), dtype=np.float32) * 3
    v_new = np.ones((1, 2, 1, 8), dtype=np.float32) * 3
    cache.update(k_new, v_new, 0)

    assert cache.get_seq_length(0) == 9
    assert cache.key_cache[0].shape[2] == 9


def test_dynamic_cache_slice_bounds():
    cache = DynamicCache()
    k = np.ones((1, 2, 25, 8), dtype=np.float32)
    v = np.ones((1, 2, 25, 8), dtype=np.float32)
    cache.update(k, v, 0)

    cache.slice(5, 15)
    assert cache.get_seq_length(0) == 10

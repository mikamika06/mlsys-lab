import numpy as np
from batchremap.cache import KVCacheBuffer
from batchremap.decode import gather_batch_kv
from batchremap.mapping import create_batch_mapping


def test_gather_batch_kv_mapping():
    """Test batch KV gather and index mapping logic."""
    active_requests = ["req_A", "req_B"]
    req_to_cache_slot = {"req_A": 3, "req_B": 1}
    max_cache_batch = 5

    cache_batch_idx = create_batch_mapping(active_requests, req_to_cache_slot, max_cache_batch)
    assert np.array_equal(cache_batch_idx, np.array([3, 1], dtype=np.int32))

    kv_cache = KVCacheBuffer(max_cache_batch=5, max_seq_len=10, num_heads=2, head_dim=4)
    seq_lens = np.array([2, 1], dtype=np.int32)
    new_k = np.ones((2, 2, 4), dtype=np.float32)
    new_v = np.ones((2, 2, 4), dtype=np.float32) * 2.0

    kv_cache.update_and_fetch(cache_batch_idx, seq_lens, new_k, new_v)

    k_out, v_out = gather_batch_kv(kv_cache, cache_batch_idx, seq_lens + 1)
    assert k_out.shape == (2, 3, 2, 4)
    assert v_out.shape == (2, 3, 2, 4)
    assert np.allclose(k_out[0, 2], new_k[0])
    assert np.allclose(k_out[1, 1], new_k[1])
    assert np.allclose(k_out[1, 2], 0.0)

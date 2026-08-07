import sys
import numpy as np

sys.path.insert(0, ".")
from kvcache.offloaded import DynamicCache, OffloadedCache

def test_offloaded_cache_equivalence():
    dc = DynamicCache()
    oc = OffloadedCache()

    np.random.seed(42)
    for layer_idx in range(2):
        k1 = np.random.randn(1, 4, 8)
        v1 = np.random.randn(1, 4, 8)

        dc_k, dc_v = dc.update(k1, v1, layer_idx)
        oc_k, oc_v = oc.update(k1, v1, layer_idx)

        assert np.array_equal(dc_k, oc_k), f"Keys differ at layer {layer_idx}"
        assert np.array_equal(dc_v, oc_v), f"Values differ at layer {layer_idx}"
        assert dc.get_seq_length(layer_idx) == oc.get_seq_length(layer_idx)

        k2 = np.random.randn(1, 3, 8)
        v2 = np.random.randn(1, 3, 8)

        dc_k, dc_v = dc.update(k2, v2, layer_idx)
        oc_k, oc_v = oc.update(k2, v2, layer_idx)

        assert np.array_equal(dc_k, oc_k), f"Keys differ after append at layer {layer_idx}"
        assert np.array_equal(dc_v, oc_v), f"Values differ after append at layer {layer_idx}"
        assert dc.get_seq_length(layer_idx) == oc.get_seq_length(layer_idx)

def test_offloaded_cache_seq_length():
    oc = OffloadedCache()
    k1 = np.random.randn(1, 4, 8)
    v1 = np.random.randn(1, 4, 8)
    oc.update(k1, v1, 0)
    assert oc.get_seq_length(0) == 4
    assert oc.get_seq_length(1) == 0

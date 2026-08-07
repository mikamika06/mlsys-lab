import sys
import numpy as np

sys.path.insert(0, ".")
from kvcache.config import LayerConfig, ModelConfig
from kvcache.cache import HybridKVCache


def test_sliding_window_evicts_oldest():
    cfg = ModelConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        layer_configs=[LayerConfig(layer_id=0, is_sliding=True, window_size=3)],
    )
    cache = HybridKVCache(cfg)
    rng = np.random.default_rng(123)

    last_k = None
    for i in range(6):
        k = rng.standard_normal((1, 2, 4)).astype(np.float32)
        v = rng.standard_normal((1, 2, 4)).astype(np.float32)
        if i == 5:
            last_k = k
        cache.update(0, k, v)

    k_cached, _ = cache.get_kv(0)
    assert k_cached.shape[0] == 3
    assert np.allclose(k_cached[-1], last_k[0])


def test_full_layer_does_not_truncate():
    cfg = ModelConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        layer_configs=[LayerConfig(layer_id=0, is_sliding=False)],
    )
    cache = HybridKVCache(cfg)
    rng = np.random.default_rng(456)

    for _ in range(25):
        k = rng.standard_normal((1, 2, 4)).astype(np.float32)
        v = rng.standard_normal((1, 2, 4)).astype(np.float32)
        cache.update(0, k, v)

    k_cached, _ = cache.get_kv(0)
    assert k_cached.shape[0] == 25

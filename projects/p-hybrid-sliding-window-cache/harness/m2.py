import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from kvcache.cache import HybridKVCache

    m = {
        "sliding_layer_capped": 0.0,
        "full_layer_unbounded": 0.0,
        "kv_contents_correct": 0.0,
    }

    try:
        cfg = ref.make_sample_config(num_layers=2, window_size=4)
        cache = HybridKVCache(cfg)

        q, k, v = ref.generate_synthetic_data(10, cfg.num_heads, cfg.head_dim)

        for t in range(10):
            k_step = k[t : t + 1]
            v_step = v[t : t + 1]
            cache.update(0, k_step, v_step)
            cache.update(1, k_step, v_step)

        k_full, v_full = cache.get_kv(0)
        k_slide, v_slide = cache.get_kv(1)

        if k_full.shape[0] == 10 and v_full.shape[0] == 10:
            m["full_layer_unbounded"] = 1.0

        if k_slide.shape[0] == 4 and v_slide.shape[0] == 4:
            m["sliding_layer_capped"] = 1.0

        if np.allclose(k_slide, k[6:10]) and np.allclose(v_slide, v[6:10]):
            m["kv_contents_correct"] = 1.0
    except Exception:
        pass

    return m

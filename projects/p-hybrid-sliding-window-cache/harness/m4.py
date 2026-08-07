import sys


def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from kvcache.cache import HybridKVCache

    m = {"memory_measured": 0.0, "saving_ratio_correct": 0.0}

    try:
        cfg = ref.make_sample_config(num_layers=4, window_size=10)
        cache = HybridKVCache(cfg)

        q, k, v = ref.generate_synthetic_data(50, cfg.num_heads, cfg.head_dim)

        for t in range(50):
            for l_idx in range(4):
                cache.update(l_idx, k[t : t + 1], v[t : t + 1])

        alloc = cache.total_allocated_slots()
        naive = cache.naive_allocated_slots()
        ratio = cache.memory_saving_ratio()

        if alloc == 120 and naive == 200:
            m["memory_measured"] = 1.0

        expected_ratio = (200.0 - 120.0) / 200.0
        if abs(ratio - expected_ratio) < 1e-6:
            m["saving_ratio_correct"] = 1.0
    except Exception:
        pass

    return m

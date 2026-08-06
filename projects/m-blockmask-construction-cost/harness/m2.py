import ref


def check(workdir):
    from flexmask.cost import BlockMaskCostProfiler
    from flexmask.cache import MaskCache, DummyBlockMask

    out = {
        "cache_hits_correct": 0.0,
        "cache_eviction_correct": 0.0,
        "amortized_latency_ratio": 0.0,
    }

    cache = MaskCache(max_capacity=2)

    def builder_a():
        return DummyBlockMask((2048, 2048), 128, 100, "causal")

    def builder_b():
        return DummyBlockMask((4096, 4096), 128, 400, "causal")

    def builder_c():
        return DummyBlockMask((8192, 8192), 128, 1600, "causal")

    m_a1, hit_a1 = cache.get_or_create((2048, 2048), 128, "causal", builder_a)
    m_a2, hit_a2 = cache.get_or_create((2048, 2048), 128, "causal", builder_a)

    if not hit_a1 and hit_a2 and m_a1 is m_a2:
        out["cache_hits_correct"] = 1.0

    cache.get_or_create((4096, 4096), 128, "causal", builder_b)
    cache.get_or_create((8192, 8192), 128, "causal", builder_c)

    m_a3, hit_a3 = cache.get_or_create((2048, 2048), 128, "causal", builder_a)
    if not hit_a3 and cache.size() == 2:
        out["cache_eviction_correct"] = 1.0

    profiler = BlockMaskCostProfiler(block_size=128)
    sim = profiler.simulate_flex_vs_fa2_latency(seq_len=4096, num_heads=32)
    amortized_flex_latency = sim["flex_kernel_us"]
    amortized_ratio = amortized_flex_latency / max(1e-6, sim["fa2_latency_us"])

    out["amortized_latency_ratio"] = float(amortized_ratio)
    return out

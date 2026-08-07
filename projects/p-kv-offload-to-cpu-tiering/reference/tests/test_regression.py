from kvtier.tier import TieredStorage


def test_cpu_tier_full_rejection():
    ts = TieredStorage(gpu_capacity=1, cpu_capacity=1)
    assert ts.access("s1") == "gpu"
    assert ts.evict_to_cpu("s1") is True
    assert ts.access("s2") == "gpu"
    assert ts.evict_to_cpu("s2") is False

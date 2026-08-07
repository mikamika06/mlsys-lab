import sys

sys.path.insert(0, ".")
from kvcalc.calc import calculate_concurrency, effective_capacity


def test_concurrency_within_limits():
    cfg = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2}
    workload = {"total_bytes": 16 * 1024**3, "block_size": 16, "avg_seq_len": 512, "burst_factor": 1.2}
    res = calculate_concurrency(cfg, workload)
    assert res > 0, "Concurrency must be positive"
    assert res < 100000, "Concurrency unreasonably high"


def test_effective_capacity_non_negative():
    cap = effective_capacity(1024**3, 16, 100)
    assert cap >= 0, "Capacity cannot be negative"

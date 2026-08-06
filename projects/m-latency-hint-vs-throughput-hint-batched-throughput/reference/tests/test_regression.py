import sys

sys.path.insert(0, ".")
from cpuhints.scheduler import estimate_throughput


def test_throughput_hint_scales_better_on_large_batches():
    cores = 16
    batch_sizes = [64]

    lat = estimate_throughput(batch_sizes, "latency", cores)[64]
    thr = estimate_throughput(batch_sizes, "throughput", cores)[64]

    assert thr > lat, f"Throughput hint ({thr}) should outscale latency hint ({lat}) for large batches"

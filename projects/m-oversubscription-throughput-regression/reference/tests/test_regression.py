import sys
sys.path.insert(0, ".")
from oversub.scheduler import OptimalStreamPool


def dummy_bench(streams):
    num_cores = 4
    if streams <= num_cores:
        return float(streams * 100)
    return float((num_cores * 100) / (1.0 + 0.5 * (streams - num_cores)))


def test_knee_point_prevents_oversubscription():
    pool = OptimalStreamPool(dummy_bench, max_streams=16)
    opt = pool.get_optimal_streams()
    assert opt <= 4, f"Optimal streams {opt} exceeds core capacity 4"
    assert opt == 4, f"Optimal streams expected 4, got {opt}"
    ratio = pool.compute_throughput_ratio(16)
    assert ratio > 1.0, f"Expected throughput ratio > 1.0, got {ratio}"

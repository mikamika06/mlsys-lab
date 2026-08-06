import sys
sys.path.insert(0, ".")
from fsdpmeasure.memory import simulate_peak_memory
from fsdpmeasure.residency import get_parameter_residency
from fsdpmeasure.overhead import compute_gloo_overhead

def test_simulate_peak_memory_sanity():
    m = simulate_peak_memory(100000000, 4, "FULL_SHARD")
    assert len(m) == 4
    assert m[0] > 0

def test_get_parameter_residency_diff():
    full = get_parameter_residency(1000000, "FULL_SHARD", "between_forward")
    shard_grad = get_parameter_residency(1000000, "SHARD_GRAD_OP", "between_forward")
    assert full < shard_grad

def test_compute_gloo_overhead_positive():
    t = compute_gloo_overhead(1.0, 10, 1024)
    assert t > 1.0

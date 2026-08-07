import sys
import numpy as np

sys.path.insert(0, ".")
from zero_diag.parser import parse_memory_estimator_log
from zero_diag.fragmentation import compute_fragmentation_curve
from zero_diag.planner import compute_optimal_reduce_bucket_size

def test_parser_valid_output():
    log = """
    DeepSpeed ZeRO-2 Memory Estimator
    Total Number of Parameters: 7_000_000_000
    ZeRO Stage: 2
    World Size: 8
    Base Memory: 14.00 GB
    Gradient Memory: 3.50 GB
    Total Memory: 17.50 GB
    """
    parsed = parse_memory_estimator_log(log)
    assert parsed["params_numel"] == 7000000000
    assert parsed["zero_stage"] == 2
    assert parsed["world_size"] == 8
    assert parsed["total_mem_gb"] == 17.50

def test_fragmentation_non_zero():
    shapes = [(1, 3), (7, 13), (100, 105)]
    res = compute_fragmentation_curve(shapes, elem_bytes=2, alignment_bytes=512)
    assert res["total_overhead_bytes"] > 0
    assert len(res["fragmentation_ratio_curve"]) == 3
    assert res["cumulative_padded_bytes"][-1] >= res["cumulative_exact_bytes"][-1]

def test_optimal_bucket_within_bounds():
    total_params = 100_000_000
    opt_bucket = compute_optimal_reduce_bucket_size(
        total_params=total_params,
        num_ranks=16,
        latency_sec=1e-5,
        bandwidth_bytes_per_sec=1e10,
        elem_bytes=2,
        max_mem_bytes=50_000_000
    )
    assert opt_bucket > 0
    assert opt_bucket * 2 <= 50_000_000
    assert opt_bucket <= total_params

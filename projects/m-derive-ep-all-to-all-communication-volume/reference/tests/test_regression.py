import pytest
from epall.volume import compute_ep_all_to_all_volume
from epall.log import reconstruct_token_counts
from epall.bench import measure_ep_vs_tp_throughput

def test_volume_positive():
    val = compute_ep_all_to_all_volume(8, 2048, 4096, 2, 2)
    assert val > 0

def test_log_reconstruction():
    logs = [{"src": 0, "dst": 1, "count": 10}]
    mat = reconstruct_token_counts(logs, 2)
    assert mat[0, 1] == 10

def test_bench_ratio():
    ratio = measure_ep_vs_tp_throughput([10.0], [20.0])
    assert ratio == 2.0

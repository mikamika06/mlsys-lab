import pytest
from shm.bytes import compute_shm_bytes
from shm.sweep import analyze_sweep


def test_compute_shm_basic():
    res = compute_shm_bytes(64, 64, 32, 3, "float16")
    assert res > 0
    assert isinstance(res, int)


def test_sweep_analysis():
    recs = [
        {"num_warps": 4, "achieved_occupancy": 0.5},
        {"num_warps": 8, "achieved_occupancy": 0.8},
    ]
    res = analyze_sweep(recs)
    assert res["best_num_warps"] == 8
    assert res["max_occupancy"] == 0.8

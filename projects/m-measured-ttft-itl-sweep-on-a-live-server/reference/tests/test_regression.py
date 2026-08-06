import sys
sys.path.insert(0, ".")
from sweep.alignment import align_chunks
from sweep.metrics import compute_blocking
from sweep.server import simulate_sweep

def test_boundary_alignment_exact():
    assert align_chunks(512, 16) == 512

def test_boundary_alignment_rounding():
    assert align_chunks(500, 16) == 512

def test_blocking_metric_positive():
    val = compute_blocking([2.0, 5.0, 20.0], 128)
    assert val > 0.0

def test_simulate_sweep_length():
    res = simulate_sweep({"steps": 5, "base_ttft": 10.0, "base_itl": 2.0})
    assert len(res) == 5

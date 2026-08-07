import sys
sys.path.insert(0, ".")
from arena.passes import track_arena_sizes
from arena.separation import compute_separation_savings
from arena.detect import detect_dynamic_tensors


def test_arena_convergence():
    passes = [{"arena_size": 1024}, {"arena_size": 512}, {"arena_size": 512}]
    res = track_arena_sizes(passes)
    assert res["converged"] is True
    assert res["max_size"] == 1024


def test_separation_savings_positive():
    tensors = [{"size": 100, "is_constant": True}, {"size": 200, "is_constant": False}]
    res = compute_separation_savings(tensors, "segmented", "isolated")
    assert res["savings_a"] >= 0
    assert res["savings_b"] >= 0


def test_unplanned_tensor_detection():
    prog = {"nodes": [{"outputs": [{"name": "t1", "shape": [1, "N"]}]}]}
    unplanned = detect_dynamic_tensors(prog)
    assert "t1" in unplanned

import sys
sys.path.insert(0, ".")
from batch.builder import build_arguments
from batch.offsets import compute_offsets
from batch.latency import analytic_latency

REQUESTS = [
    {"id": 1, "tokens": [10, 20, 30]},
    {"id": 2, "tokens": [40, 50]}
]
HW = {"bandwidth": 1000.0, "flops": 10000.0}

def test_arguments_structure():
    args = build_arguments(REQUESTS)
    assert "cu_seqlens" in args
    assert args["max_seqlen"] == 3
    assert args["batch_size"] == 2

def test_offsets_correctness():
    args = build_arguments(REQUESTS)
    offs = compute_offsets(args)
    assert len(offs) == 2
    assert offs[0][0] == 0
    assert offs[0][1] == 3
    assert offs[1][0] == 3
    assert offs[1][1] == 5

def test_latency_model():
    args = build_arguments(REQUESTS)
    lat = analytic_latency(args, HW)
    assert lat["ttft"] > 0
    assert lat["itl"] > 0

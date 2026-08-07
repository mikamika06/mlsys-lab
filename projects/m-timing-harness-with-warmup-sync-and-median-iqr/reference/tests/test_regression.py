import sys

sys.path.insert(0, ".")
from benchmark import benchmark_step


def test_warmup_and_reps_executed():
    state = {"calls": 0}

    def dummy():
        state["calls"] += 1

    benchmark_step(dummy, is_cuda=False, warmup=8, reps=12)
    assert state["calls"] == 20, f"Expected 20 calls, got {state['calls']}"

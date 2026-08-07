import sys

sys.path.insert(0, ".")
from preemption.simulator import simulate_wasted_tokens
from preemption.model import find_breakeven_point
from preemption.replay import select_victim


def test_simulator_basic():
    reqs = [{"id": 1, "prefill_len": 100}, {"id": 2, "prefill_len": 200}]
    preempts = [{"request_id": 1, "step": 5}]
    assert simulate_wasted_tokens(reqs, preempts) == 100


def test_model_breakeven():
    be = find_breakeven_point(1024, 10.0, 0.05)
    assert be > 0


def test_replay_policy():
    reqs = [
        {"id": 1, "arrival_time": 1, "last_active_time": 10, "prefill_len": 128},
        {"id": 2, "arrival_time": 2, "last_active_time": 5, "prefill_len": 512}
    ]
    assert select_victim(reqs, "lru") == 2
    assert select_victim(reqs, "longest_prefill") == 2

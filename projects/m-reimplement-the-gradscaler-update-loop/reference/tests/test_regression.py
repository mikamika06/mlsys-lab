import sys

sys.path.insert(0, ".")
from scaler.core import find_safe_scale, update_scaler
from scaler.trajectory import simulate_trajectory


def test_scaler_invariants():
    state = {
        "scale": 1024.0,
        "growth_track": 0,
        "growth_interval": 10,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
    }
    for _ in range(10):
        update_scaler(state, False)
    assert state["scale"] == 2048.0, f"scale expected 2048.0, got {state['scale']}"


def test_inf_backoff():
    grads = [[1.0, float("nan")], [3.0, 4.0]]
    scale, has_inf = find_safe_scale(grads, 1024.0)
    assert has_inf is True
    assert scale == 512.0


def test_trajectory_length():
    history = [[[1.0, 2.0]] for _ in range(5)]
    res = simulate_trajectory(history, initial_scale=100.0, growth_interval=2)
    assert len(res) == 5

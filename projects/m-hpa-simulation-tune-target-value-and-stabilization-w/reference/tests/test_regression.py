import sys

sys.path.insert(0, ".")
from hpa.simulator import simulate_hpa

def test_stabilization_window_prevents_premature_scale_down():
    metrics = [10.0, 100.0, 1.0, 1.0]
    reps = simulate_hpa(metrics, 1, 10.0, 3)

    assert reps[2] == 10, f"Expected replicas to scale to 10, got {reps[2]}"
    assert reps[3] == 10, f"Stabilization window should hold replicas at 10, got {reps[3]}"

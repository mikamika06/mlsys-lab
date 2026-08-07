import sys
sys.path.insert(0, ".")
from kbitbug.reproduce import simulate_training_step
from kbitbug.packing import compute_token_utilization
from kbitbug.overfit import find_overfitting_step
import numpy as np

def test_simulation_zero_gradients():
    np.random.seed(0)
    inputs = np.random.randn(5, 3)
    weights = np.random.randn(3, 2)
    targets = np.random.randn(5, 2)
    _, grad_sk = simulate_training_step(weights, inputs, targets, skipped_preparation=True)
    _, grad_ok = simulate_training_step(weights, inputs, targets, skipped_preparation=False)
    assert np.all(grad_sk == 0.0)
    assert not np.all(grad_ok == 0.0)

def test_packing_utilization_values():
    lengths = [10, 20, 30]
    res = compute_token_utilization(lengths, 100)
    assert res["actual_tokens"] == 60
    assert res["packing_total"] == 100

def test_overfitting_step_detection():
    logs = [
        {"step": 10, "eval_loss": 2.5},
        {"step": 20, "eval_loss": 1.8},
        {"step": 30, "eval_loss": 2.1}
    ]
    assert find_overfitting_step(logs) == 20

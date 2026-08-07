import sys
sys.path.insert(0, ".")
import numpy as np
from recovery.trainer import compute_kd_loss
from recovery.eval import steps_to_90_recovery


def test_kd_loss_non_zero():
    X = np.random.randn(10, 4)
    s = np.random.randn(4, 2)
    t = np.random.randn(4, 2)
    loss = compute_kd_loss(s, t, X)
    assert loss > 0.0, f"KD loss expected positive, got {loss}"


def test_steps_to_recovery_bounds():
    accs = [0.1, 0.2, 0.5, 0.9, 0.95]
    steps = steps_to_90_recovery(accs, baseline_acc=1.0, pruned_acc=0.0)
    assert steps >= 0
    assert steps < len(accs)

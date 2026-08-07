import sys
import numpy as np

sys.path.insert(0, ".")
from spectrain.teacher import teacher_forced_loss
from spectrain.acceptance import compute_acceptance_rate
from spectrain.objective import combined_objective


def test_teacher_loss_non_negative():
    tokens = [1, 2, 3, 4, 5]
    logits = np.zeros((5, 10), dtype=np.float32)
    loss = teacher_forced_loss(tokens, logits)
    assert loss >= 0.0


def test_acceptance_rate_bounds():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    draft_logits = np.zeros((8, 10), dtype=np.float32)
    target_logits = np.zeros((8, 10), dtype=np.float32)
    rate = compute_acceptance_rate(tokens, draft_logits, target_logits, gamma=2)
    assert 0.0 <= rate <= 1.0


def test_combined_objective_returns_float():
    tokens = [1, 2, 3, 4, 5]
    draft_logits = np.zeros((5, 10), dtype=np.float32)
    target_logits = np.zeros((5, 10), dtype=np.float32)
    obj = combined_objective(tokens, draft_logits, target_logits, gamma=2)
    assert isinstance(obj, float)

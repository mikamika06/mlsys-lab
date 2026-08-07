import sys
import numpy as np
sys.path.insert(0, ".")
from draftdistill.loss import combined_loss
from draftdistill.evaluation import compute_acceptance_rate, evaluate_dataset_sizes

def test_combined_loss_non_negative():
    logits = np.zeros((10, 5))
    targets = np.zeros((10, 5))
    f_pred = np.zeros((10, 8))
    f_targ = np.zeros((10, 8))
    loss = combined_loss(logits, targets, f_pred, f_targ)
    assert loss >= 0.0

def test_acceptance_rate_bounds():
    logits = np.random.randn(20, 5)
    tokens = np.random.randint(0, 5, size=(20,))
    rate = compute_acceptance_rate(logits, tokens)
    assert 0.0 <= rate <= 1.0

def test_scaling_trend():
    res = evaluate_dataset_sizes([100, 500, 2000])
    assert res[2000] > res[100]

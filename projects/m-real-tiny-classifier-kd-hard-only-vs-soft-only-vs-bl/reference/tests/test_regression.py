import sys
sys.path.insert(0, ".")
import numpy as np
from kd.loss import hard_loss, soft_loss, blended_loss
from kd.train import run_epoch, evaluate_accuracy


def test_loss_non_negative():
    logits = np.array([[2.0, 1.0, 0.1], [0.1, 2.5, 0.3]])
    targets = np.array([0, 1])
    assert hard_loss(logits, targets) >= 0.0


def test_soft_loss_temperature_scaling():
    s_logits = np.array([[1.0, 2.0], [2.0, 1.0]])
    t_logits = np.array([[1.5, 2.5], [2.5, 1.5]])
    loss_t1 = soft_loss(s_logits, t_logits, 1.0)
    loss_t4 = soft_loss(s_logits, t_logits, 4.0)
    assert loss_t1 >= 0.0
    assert loss_t4 >= 0.0


def test_blended_weights_sum():
    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    targets = np.array([0, 1])
    s_logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    t_logits = np.array([[1.2, -0.2], [-0.2, 1.2]])
    bl = blended_loss(logits, targets, s_logits, t_logits, alpha=0.5, temperature=2.0)
    assert bl >= 0.0


def test_evaluate_accuracy_bounds():
    logits = np.array([[10.0, 0.0], [0.0, 10.0]])
    targets = np.array([0, 1])
    assert evaluate_accuracy(logits, targets) == 1.0

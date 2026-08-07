import sys
sys.path.insert(0, ".")
from redscale.nonassoc import quantify_non_associativity
from redscale.outliers import detect_loss_spike_ranks
from redscale.overflow import global_overflow_skip
import numpy as np


def test_non_associativity_non_negative():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    res = quantify_non_associativity(arr)
    assert res >= 0.0


def test_outlier_detection_shape():
    mat = np.array([[1.0, 1.1], [1.0, 10.0]], dtype=np.float32)
    spikes = detect_loss_spike_ranks(mat, multiplier=2.0)
    assert isinstance(spikes, list)


def test_overflow_skip_logic():
    flags = [False, True, False]
    assert global_overflow_skip(flags) is True
    assert global_overflow_skip([False, False]) is False

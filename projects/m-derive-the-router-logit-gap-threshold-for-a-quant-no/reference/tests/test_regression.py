import numpy as np
import pytest
from routerquant.threshold import derive_logit_gap_threshold


def test_threshold_properties():
    logits = np.array([[10.0, 8.0, 5.0]])
    weights = np.zeros((3, 3))
    quant_weights = np.ones((3, 3)) * 0.1
    hidden = np.ones((1, 3))
    thresh = derive_logit_gap_threshold(logits, weights, quant_weights, hidden)
    assert thresh.shape == (1,)
    assert thresh[0] >= 0.0


def test_threshold_value():
    logits = np.array([[12.0, 9.0, 4.0]])
    weights = np.zeros((3, 3))
    quant_weights = np.zeros((3, 3))
    hidden = np.ones((1, 3))
    thresh = derive_logit_gap_threshold(logits, weights, quant_weights, hidden)
    assert np.isclose(thresh[0], 3.0)

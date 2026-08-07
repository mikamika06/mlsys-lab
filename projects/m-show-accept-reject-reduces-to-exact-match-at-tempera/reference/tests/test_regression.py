import sys
sys.path.insert(0, ".")
from specsampling.core import verify_zero_temp_reduction, measure_acceptance_rates, quantify_mismatch_skew
import numpy as np

def test_zero_temperature_acceptance_is_deterministic():
    t_logits = np.array([2.0, 1.0, 0.0])
    d_logits = np.array([2.5, 0.5, 0.1])
    assert verify_zero_temp_reduction(t_logits, d_logits) is True

def test_acceptance_rate_bounds():
    t_logits = np.zeros(10)
    d_logits = np.zeros(10)
    rates = measure_acceptance_rates(t_logits, d_logits, [1.0])
    assert 0.9 <= rates[0] <= 1.0

def test_temperature_mismatch_detection():
    t_logits = np.array([1.0, 2.0, 5.0])
    d_logits = np.array([5.0, 2.0, 1.0])
    skew = quantify_mismatch_skew(t_logits, d_logits, 0.5, 1.0)
    assert skew > 0.05

import sys
import numpy as np

sys.path.insert(0, ".")
from distill.diagnose import derive_effective_temperature, detect_mode_collapse


def test_effective_temperature_scaling():
    logits = np.array([[5.0, 0.1, 0.1], [6.0, 0.2, 0.1]])
    t_eff_high = derive_effective_temperature(logits, target_temperature=2.0, confidence_alpha=1.0)
    t_eff_zero = derive_effective_temperature(logits, target_temperature=2.0, confidence_alpha=0.0)

    assert t_eff_high > t_eff_zero, "Overconfident teacher must increase effective temperature"
    assert t_eff_zero == 2.0, "Zero alpha must return baseline temperature"


def test_mode_collapse_detection_sensitivity():
    collapsed_batch = np.array([[10.0, -10.0, -10.0], [12.0, -8.0, -8.0]])
    diverse_batch = np.array([[1.0, 0.9, 0.8], [0.5, 0.6, 0.4]])

    history = [diverse_batch, collapsed_batch]
    detected = detect_mode_collapse(history, entropy_threshold=0.3)

    assert 1 in detected, "Step 1 must be flagged as mode collapse"
    assert 0 not in detected, "Step 0 must not be flagged as mode collapse"

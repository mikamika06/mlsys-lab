import sys
sys.path.insert(0, ".")
from mtpcalc.heads import compute_second_position_accuracy
from mtpcalc.multiplier import compute_tps_multiplier

def test_second_position_accuracy_range():
    logits = [[[1.0, 0.0], [0.0, 1.0]]]
    targets = [0, 1]
    acc = compute_second_position_accuracy("sequential", logits, targets)
    assert 0.0 <= acc <= 1.0, f"accuracy {acc} out of range [0, 1]"

def test_multiplier_positive():
    m = compute_tps_multiplier(0.8)
    assert m > 1.0, f"multiplier {m} should be > 1.0"

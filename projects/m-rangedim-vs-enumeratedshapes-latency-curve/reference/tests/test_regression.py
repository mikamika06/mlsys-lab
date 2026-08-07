import sys
sys.path.insert(0, ".")
from shapes.evaluator import evaluate_rangedim, evaluate_enumerated
from shapes.enumerator import minimal_shape_set
from shapes.curve import latency_ratio_curve


def test_numerics_identical():
    inputs = [16, 32, 64, 128]
    r_out = evaluate_rangedim(("seq", 16, 64, 128), inputs)
    e_out = evaluate_enumerated([16, 64, 128], inputs)
    assert len(r_out) == len(e_out)
    for a, b in zip(r_out, e_out):
        assert abs(a - b) < 1e-5


def test_minimal_shape_set_non_empty():
    hist = {10: 5, 20: 10, 50: 2, 100: 1}
    res = minimal_shape_set(hist, max_waste=0.2)
    assert len(res) > 0
    assert res[0] == 10


def test_latency_ratio_structure():
    ranges = [10, 20, 50]
    profile = {("rangedim", 10): 1.5, ("enumerated", 10): 1.0,
               ("rangedim", 20): 2.0, ("enumerated", 20): 1.2,
               ("rangedim", 50): 3.0, ("enumerated", 50): 2.0}
    curve = latency_ratio_curve(ranges, profile)
    assert len(curve) == 3
    for length, ratio in curve:
        assert ratio > 0.0

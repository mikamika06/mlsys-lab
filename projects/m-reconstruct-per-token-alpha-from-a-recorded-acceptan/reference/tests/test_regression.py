import sys
sys.path.insert(0, ".")
from specalpha.reconstruct import reconstruct_alphas
from specalpha.metrics import expected_speedup
from specalpha.utils import normalize_histogram


def test_alphas_bounds():
    hist = {0: 10, 1: 20, 2: 30, 3: 40}
    alphas = reconstruct_alphas(hist, 3)
    for a in alphas:
        assert 0.0 <= a <= 1.0, f"alpha {a} out of bounds"


def test_expected_speedup_basic():
    alphas = [0.5, 0.5, 0.5]
    s = expected_speedup(alphas)
    assert s > 0.0, "speedup must be positive"


def test_normalize_sums_to_one():
    hist = {0: 5, 1: 15, 2: 30}
    norm = normalize_histogram(hist)
    total = sum(norm.values())
    assert abs(total - 1.0) < 1e-5, f"normalized sum {total} != 1.0"

import sys

sys.path.insert(0, ".")
from ttft.savings import prefill_cost, simulate_batch


def test_caching_saves_compute():
    scratch = prefill_cost(100050, 0, 1e-7, 1e-5)
    cached = prefill_cost(50, 100000, 1e-7, 1e-5)
    assert cached < scratch


def test_simulate_batch_reflects_savings():
    doc = 100000
    qs = [50] * 50
    base, cached = simulate_batch(doc, qs, 1e-7, 1e-5)
    assert cached < base

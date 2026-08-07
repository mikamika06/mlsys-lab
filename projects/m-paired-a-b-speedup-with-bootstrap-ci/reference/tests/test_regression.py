import sys
sys.path.insert(0, ".")
from bench.core import paired_bootstrap_ci, robust_summary

def test_paired_bootstrap_bounds():
    a = [10.0, 10.5, 10.2, 10.1, 10.3]
    b = [5.0, 5.1, 5.05, 5.02, 5.03]
    res = paired_bootstrap_ci(a, b, n_boot=200, alpha=0.05, seed=123)
    assert res["lower"] < res["median"]
    assert res["median"] < res["upper"]

def test_robust_summary_median():
    latencies = [1.0, 2.0, 100.0]
    res = robust_summary(latencies)
    assert res["median"] == 2.0
    assert res["mean"] > res["median"]

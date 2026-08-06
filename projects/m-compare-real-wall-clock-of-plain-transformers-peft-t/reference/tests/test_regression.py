"""Regression tests for framework benchmarking."""

from bench.profile import profile_runtimes
from bench.loss import verify_loss_convergence

def test_profile_sync_verification():
    def fast_step():
        pass

    res = profile_runtimes(fast_step, fast_step, steps=10, warmup_steps=1)
    assert res["pt_total_sec"] >= 0.0
    assert res["mlx_total_sec"] >= 0.0
    assert "latency_ratio" in res

def test_loss_parity_detector():
    pt_good = [2.5, 2.1, 1.8, 1.5, 1.2]
    mlx_good = [2.5, 2.0, 1.7, 1.48, 1.21]
    res_good = verify_loss_convergence(pt_good, mlx_good, max_relative_diff=0.1)
    assert res_good["comparable"] is True

    mlx_bad = [2.5, 2.0, 1.7, 1.48, 2.5]
    res_bad = verify_loss_convergence(pt_good, mlx_bad, max_relative_diff=0.1)
    assert res_bad["comparable"] is False

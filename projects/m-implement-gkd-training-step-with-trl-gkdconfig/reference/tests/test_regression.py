import sys
sys.path.insert(0, ".")
from gkd.step import GKDConfig, compute_gkd_step
from gkd.toy import compute_toy_divergence
from gkd.drift import measure_distribution_drift


def test_gkd_config_defaults():
    cfg = GKDConfig()
    assert cfg.beta == 0.0
    assert cfg.lmbda == 1.0
    assert cfg.temperature == 1.0


def test_compute_gkd_step_output_keys():
    cfg = GKDConfig(beta=0.5)
    res = compute_gkd_step([[1.0, 2.0]], [[1.5, 1.0]], cfg)
    assert "loss" in res
    assert "grad_norm" in res
    assert isinstance(res["loss"], float)


def test_toy_divergence_non_negative():
    p = [2.0, 3.0, 5.0]
    q = [1.0, 4.0, 5.0]
    for mode in ["forward_kl", "reverse_kl", "jsd"]:
        val = compute_toy_divergence(p, q, mode=mode)
        assert val >= -1e-7, f"Divergence {mode} negative: {val}"


def test_distribution_drift_bounds():
    on_p = [[0.8, 0.2], [0.5, 0.5]]
    off_p = [[0.1, 0.9], [0.5, 0.5]]
    drift = measure_distribution_drift(on_p, off_p)
    assert 0.0 <= drift["tv_distance"] <= 1.0
    assert drift["kl_drift"] >= 0.0

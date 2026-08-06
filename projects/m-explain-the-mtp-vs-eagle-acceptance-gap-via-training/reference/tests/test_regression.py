import sys
sys.path.insert(0, ".")
import numpy as np
from mtpgap.model import compute_mtp_loss, estimate_gradient_interference
from mtpgap.simulation import simulate_acceptance_rates, compute_trajectory_divergence
from mtpgap.analysis import analyze_gap


def test_mtp_loss_non_negative():
    logits = [np.zeros((10, 32)), np.zeros((10, 32))]
    targets = [np.zeros(10, dtype=int), np.zeros(10, dtype=int)]
    weights = [1.0, 0.5]
    loss = compute_mtp_loss(logits, targets, weights)
    assert loss >= 0.0, "MTP loss must be non-negative"


def test_gradient_interference_bounds():
    g1 = np.array([1.0, 0.0])
    g2 = np.array([-1.0, 0.0])
    val = estimate_gradient_interference([g1, g2])
    assert -1.0 <= val <= 1.0, "Cosine similarity must be within [-1, 1]"


def test_trajectory_divergence_symmetry():
    s1 = np.random.default_rng(42).normal(size=(5, 64))
    s2 = np.random.default_rng(43).normal(size=(5, 64))
    d1 = compute_trajectory_divergence(s1, s2)
    d2 = compute_trajectory_divergence(s2, s1)
    assert abs(d1 - d2) < 1e-6, "Trajectory divergence must be symmetric"


def test_analysis_gap_keys():
    config = {
        "logits": [np.zeros((4, 16)), np.zeros((4, 16))],
        "targets": [np.zeros(4, dtype=int), np.zeros(4, dtype=int)],
        "weights": [1.0, 1.0],
        "mtp_probs": np.ones((4, 16)) / 16.0,
        "eagle_probs": np.ones((4, 16)) / 16.0,
        "temperature": 1.0
    }
    res = analyze_gap(config)
    assert "gap" in res, "Analysis result must include gap"

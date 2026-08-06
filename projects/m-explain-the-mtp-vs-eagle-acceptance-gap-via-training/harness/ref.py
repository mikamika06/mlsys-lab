import numpy as np
from mtpgap.analysis import analyze_gap
from mtpgap.model import compute_mtp_loss, compute_eagle_loss, estimate_gradient_interference
from mtpgap.simulation import simulate_acceptance_rates, compute_trajectory_divergence

CONFIGS = [
    {
        "logits": [np.zeros((8, 24)), np.zeros((8, 24))],
        "targets": [np.zeros(8, dtype=int), np.zeros(8, dtype=int)],
        "weights": [1.0, 0.8],
        "mtp_probs": np.ones((8, 24)) / 24.0,
        "eagle_probs": np.ones((8, 24)) / 24.0,
        "temperature": 1.0
    },
    {
        "logits": [np.ones((8, 24)), np.ones((8, 24))],
        "targets": [np.ones(8, dtype=int), np.ones(8, dtype=int)],
        "weights": [1.0, 0.5],
        "mtp_probs": np.ones((8, 24)) / 24.0,
        "eagle_probs": np.ones((8, 24)) / 24.0,
        "temperature": 0.8
    },
    {
        "logits": [np.random.default_rng(42).normal(size=(8, 24)), np.random.default_rng(43).normal(size=(8, 24))],
        "targets": [np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([7, 6, 5, 4, 3, 2, 1, 0])],
        "weights": [1.0, 0.9],
        "mtp_probs": np.ones((8, 24)) / 24.0,
        "eagle_probs": np.ones((8, 24)) / 24.0,
        "temperature": 1.2
    }
]

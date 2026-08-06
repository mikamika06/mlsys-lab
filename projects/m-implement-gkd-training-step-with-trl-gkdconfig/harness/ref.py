import numpy as np
from reference.gkd.step import GKDConfig, compute_gkd_step
from reference.gkd.toy import compute_toy_divergence
from reference.gkd.drift import measure_distribution_drift

CONFIGS = [
    GKDConfig(beta=0.0, temperature=1.0),
    GKDConfig(beta=1.0, temperature=1.2),
    GKDConfig(beta=0.5, temperature=0.8),
]

TEST_CASES_STEP = [
    ([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]], [[1.2, 1.8, 2.5], [0.0, 0.0, 1.0]], cfg)
    for cfg in CONFIGS
]

TEST_CASES_TOY = [
    ([2.0, 3.0, 5.0], [1.0, 4.0, 5.0], "forward_kl", 0.0),
    ([1.0, 2.0, 7.0], [3.0, 3.0, 4.0], "reverse_kl", 0.0),
    ([2.0, 3.0, 5.0], [1.0, 4.0, 5.0], "generalized", 0.3),
]

TEST_CASES_DRIFT = [
    ([[0.8, 0.2], [0.4, 0.6]], [[0.1, 0.9], [0.5, 0.5]]),
    ([[0.9, 0.1], [0.2, 0.8]], [[0.9, 0.1], [0.2, 0.8]]),
]

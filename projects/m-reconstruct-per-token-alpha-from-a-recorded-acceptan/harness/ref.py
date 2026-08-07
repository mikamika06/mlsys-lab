import numpy as np

PROFILES = [
    {
        "max_k": 4,
        "histogram": {0: 100, 1: 250, 2: 150, 3: 75, 4: 25}
    },
    {
        "max_k": 3,
        "histogram": {0: 50, 1: 100, 2: 100, 3: 50}
    },
    {
        "max_k": 5,
        "histogram": {0: 200, 1: 300, 2: 200, 3: 100, 4: 50, 5: 20}
    }
]


def generate_reference_alphas(profile):
    from specalpha.reconstruct import reconstruct_alphas
    return reconstruct_alphas(profile["histogram"], profile["max_k"])


def generate_reference_metrics(profile):
    from specalpha.reconstruct import reconstruct_alphas
    from specalpha.metrics import expected_speedup
    alphas = reconstruct_alphas(profile["histogram"], profile["max_k"])
    return expected_speedup(alphas)

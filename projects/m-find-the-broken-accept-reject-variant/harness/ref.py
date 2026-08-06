import numpy as np
from speculative.sampling import find_correct_variant, argmin_index, accept_reject_prob
from speculative.analysis import tv_distance, compare_heuristics

TEST_CASES = [
    {
        "p": np.array([0.4, 0.3, 0.2, 0.1]),
        "q": np.array([0.3, 0.4, 0.2, 0.1]),
        "variants": [
            lambda token, p, q: 0.0,
            lambda token, p, q: 1.5,
            lambda token, p, q: accept_reject_prob(p[token], q[token])
        ]
    },
    {
        "p": np.array([0.5, 0.2, 0.2, 0.1]),
        "q": np.array([0.2, 0.5, 0.2, 0.1]),
        "variants": [
            lambda token, p, q: accept_reject_prob(p[token], q[token]),
            lambda token, p, q: p[token] / q[token],
            lambda token, p, q: 0.5
        ]
    }
]

import numpy as np


def rejection_sample_block(P: np.ndarray, Q: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """
    For every row (verification position) k, independently repeat
    n_draws times: draw x ~ Q[k], accept x with probability
    min(1, P[k,x]/Q[k,x]), otherwise resample from the residual
    distribution normalize(max(P[k]-Q[k], 0)).

    P, Q: (K, V) float64 target/draft distributions (rows sum to 1).
    n_draws: number of independent repetitions per row.
    seed: seeds np.random.default_rng for all randomness used here.

    Returns E: (K, V) float64 empirical distribution of the resulting
    output token at each position (E[k] approximates P[k]).
    """
    raise NotImplementedError('your code here')

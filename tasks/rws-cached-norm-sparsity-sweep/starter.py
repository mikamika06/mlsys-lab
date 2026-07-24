import numpy as np


def wanda_masks_for_sparsities(W: np.ndarray, X: np.ndarray, sparsities: list[float]) -> list[np.ndarray]:
    """Wanda per-row unstructured pruning masks for several sparsities,
    reusing one activation-norm pass.

    W: (out_features, in_features) float64 weight matrix.
    X: (n_samples, in_features) float64 calibration activations.
    sparsities: list of target sparsity fractions in (0, 1).

    1. col_norm[i] = ||X[:, i]||_2, computed once.
    2. S = |W| * col_norm[None, :], reused for every sparsity below.
    3. For each s in sparsities, independently per output row:
       n_prune = round(s * in_features); zero out (mask = 0) the
       n_prune lowest-scoring entries of that row (stable ascending
       argsort, take the first n_prune indices), keep (mask = 1) the
       rest.

    Returns a list of (out_features, in_features) 0/1 masks, one per
    entry of `sparsities`, in the same order.
    """
    raise NotImplementedError('your code here')

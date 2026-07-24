import numpy as np

def block_influence_ranking(
    x_in: np.ndarray,
    x_out: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """
    Compute the block influence scores and ranking for each layer.

    Parameters
    ----------
    x_in : np.ndarray
        Input activations of shape (batch, layers, features).
    x_out : np.ndarray
        Output activations of shape (batch, layers, features).

    Returns
    -------
    influences : np.ndarray
        1‑D array of length `layers` with influence scores (dtype float64).
    ranking : list[int]
        Layer indices sorted in descending order of influence.
    """
    # Normalise to avoid division by zero
    eps = 1e-12
    norms_in = np.linalg.norm(x_in, axis=2) + eps
    norms_out = np.linalg.norm(x_out, axis=2) + eps

    # Cosine similarity per sample and layer
    cos = (x_in * x_out).sum(axis=2) / (norms_in * norms_out)

    # Expected cosine over the batch
    mean_cos = cos.mean(axis=0)

    # Block influence
    influences = 1.0 - mean_cos

    # Ranking: indices sorted by descending influence
    ranking = list(np.argsort(-influences))

    return influences, ranking

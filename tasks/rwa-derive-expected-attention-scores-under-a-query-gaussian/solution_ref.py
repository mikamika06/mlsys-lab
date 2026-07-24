import numpy as np


def expected_attention_scores(queries, keys, top_k):
    queries = np.asarray(queries, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)

    d = queries.shape[1]
    mu = np.mean(queries, axis=0)
    centered = queries - mu
    cov = centered.T @ centered / (queries.shape[0] - 1)

    mean_term = keys @ mu / np.sqrt(d)
    variance_term = np.einsum("ij,jk,ik->i", keys, cov, keys) / d

    scores = mean_term + 0.5 * variance_term
    order = np.argsort(-scores, kind="stable")[:top_k]

    return scores.astype(np.float64), order.astype(np.int64)

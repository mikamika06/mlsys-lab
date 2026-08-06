import numpy as np


def compute_cheap_proxies(weights, inputs):
    """Compute Frobenius norm and diagonal Fisher proxy scores per layer."""
    acts = [inputs]
    curr = inputs
    for W in weights:
        curr = np.maximum(0, curr @ W)
        acts.append(curr)

    fro_norms = []
    diag_fishers = []

    for i, W in enumerate(weights):
        fro_norms.append(float(np.linalg.norm(W, ord="fro")))
        act = acts[i]
        act_sq = np.mean(act ** 2, axis=0)
        grad_w_approx = act_sq[:, None]
        fisher_diag = np.mean((grad_w_approx * (W ** 2)), axis=(0, 1))
        diag_fishers.append(float(fisher_diag))

    return {"frobenius": fro_norms, "diag_fisher": diag_fishers}

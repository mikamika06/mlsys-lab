import numpy as np


def check_l2_normalized(embeddings, tol=1e-5):
    norms = np.linalg.norm(embeddings, axis=-1)
    return bool(np.all(np.abs(norms - 1.0) < tol))


def analyze_model_mixing(emb_a, emb_b):
    sims = np.sum(emb_a * emb_b, axis=-1)
    return {
        "mean_dot_product": float(np.mean(sims)),
        "max_norm_a": float(np.max(np.linalg.norm(emb_a, axis=-1))),
        "max_norm_b": float(np.max(np.linalg.norm(emb_b, axis=-1))),
        "compatible": False
    }

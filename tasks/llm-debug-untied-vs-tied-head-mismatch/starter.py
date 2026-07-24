import numpy as np

def tied_head_logits(embedding_matrix):
    # TODO: use a separate head matrix instead of tying
    rng = np.random.default_rng()
    head = rng.standard_normal((embedding_matrix.shape[0], embedding_matrix.shape[1]))
    return embedding_matrix @ head.T

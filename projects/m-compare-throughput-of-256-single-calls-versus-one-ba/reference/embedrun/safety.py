import numpy as np


def validate_embedding_pipeline(embeddings, reference_embeddings):
    norms = np.linalg.norm(embeddings, axis=-1)
    if not np.all(np.abs(norms - 1.0) < 1e-4):
        raise ValueError("Embeddings are not properly L2 normalized")
    ref_norms = np.linalg.norm(reference_embeddings, axis=-1)
    if not np.all(np.abs(ref_norms - 1.0) < 1e-4):
        raise ValueError("Reference embeddings are not properly L2 normalized")
    return True

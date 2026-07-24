import numpy as np

def gather_embeddings(indices, embedding_matrix):
    indices = np.asarray(indices, dtype=np.int64)
    return np.asarray(embedding_matrix, dtype=np.float64)[indices]

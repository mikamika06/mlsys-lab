import numpy as np
from sparse_eval.pattern import compress_24_matrix, decompress_24_matrix


def save_sparse_checkpoint(matrix: np.ndarray) -> dict:
    cw, meta = compress_24_matrix(matrix)
    return {
        "compressed_weights": cw,
        "metadata": meta,
        "original_shape": matrix.shape,
        "format": "2:4_structured",
    }


def load_sparse_checkpoint(checkpoint: dict) -> np.ndarray:
    cw = checkpoint["compressed_weights"]
    meta = checkpoint["metadata"]
    shape = checkpoint["original_shape"]
    return decompress_24_matrix(cw, meta, shape[1])


def get_checkpoint_stats(matrix: np.ndarray) -> dict:
    ckpt = save_sparse_checkpoint(matrix)
    cw = ckpt["compressed_weights"]
    meta = ckpt["metadata"]
    dense_bytes = int(matrix.nbytes)
    sparse_bytes = int(cw.nbytes + meta.nbytes)
    ratio = float(sparse_bytes) / float(dense_bytes)
    return {
        "dense_bytes": dense_bytes,
        "sparse_bytes": sparse_bytes,
        "compression_ratio": ratio,
    }

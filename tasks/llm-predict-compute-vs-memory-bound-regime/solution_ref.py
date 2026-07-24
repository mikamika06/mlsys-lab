import numpy as np

def predict_regime(batch: np.ndarray,
                   seq: np.ndarray,
                   d: np.ndarray,
                   dtype: np.ndarray,
                   peak_flops: np.ndarray,
                   peak_bw: np.ndarray) -> np.ndarray:
    """
    Compute the compute‑vs memory‑bound regime for each configuration.

    Parameters
    ----------
    batch, seq : array_like of int
        Batch size and sequence length (unused in the intensity calculation).
    d : array_like of int
        Hidden dimension.
    dtype : array_like of str
        Element type names such as 'float16' or 'float32'.
    peak_flops : array_like of float
        Peak floating‑point throughput in FLOPs/s.
    peak_bw : array_like of float
        Peak memory bandwidth in Bytes/s.

    Returns
    -------
    np.ndarray[int64]
        1 if compute‑bound, 0 if memory‑bound.
    """
    d = np.asarray(d)
    dtype = np.asarray(dtype)

    bpe = np.array([np.dtype(dt).itemsize for dt in dtype], dtype=np.int64)

    flops_total = 2 * d**2 * seq
    memory_bytes = (d*d + 2*d) * bpe

    oi = flops_total / memory_bytes
    ridge = peak_flops / peak_bw

    return (oi > ridge).astype(np.int64)

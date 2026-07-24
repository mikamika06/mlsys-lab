import numpy as np

def measure_kept_tokens_and_memory(compression_ratio: float,
                                   seq_len: int,
                                   full_bytes: int) -> tuple[int, float]:
    """
    Return the number of tokens that remain after compression and the amount
    of KV memory saved.

    Parameters
    ----------
    compression_ratio : float
        Fraction of KV memory that can be discarded (0 ≤ r ≤ 1).
    seq_len : int
        Number of tokens in the sequence.
    full_bytes : int
        Size of the KV buffer before compression, in bytes.

    Returns
    -------
    kept_tokens : int
        Rounded number of tokens that are retained: round((1 - r) * seq_len).
    memory_saved : float64
        Amount of KV memory saved: r * full_bytes.
    """
    # Compute kept token count using NumPy's rounding (banker's rounding)
    kept = int(np.round((1.0 - compression_ratio) * seq_len))
    # Compute memory saved as a float64
    saved = np.float64(compression_ratio * full_bytes)
    return kept, saved

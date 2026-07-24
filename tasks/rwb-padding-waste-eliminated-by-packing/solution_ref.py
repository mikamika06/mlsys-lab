import numpy as np

def compute_padding_stats(lengths):
    """
    Compute padding statistics for a batch of sequences.

    Parameters
    ----------
    lengths : array_like of int
        One‑dimensional array containing the length of each sequence in the batch.

    Returns
    -------
    padded_tokens : int
        Total number of tokens after naïve padding (batch size times maximum length).
    packed_tokens : int
        Total number of actual tokens (sum of all lengths).
    waste_fraction : float
        Fraction of padded tokens that are wasted.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    batch = lengths.size
    max_len = int(np.max(lengths)) if batch > 0 else 0
    padded_tokens = batch * max_len
    packed_tokens = int(np.sum(lengths))
    waste_fraction = (padded_tokens - packed_tokens) / padded_tokens if padded_tokens > 0 else 0.0
    return padded_tokens, packed_tokens, waste_fraction

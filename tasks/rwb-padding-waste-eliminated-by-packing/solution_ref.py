def compute_padding_stats(lengths: list[int]) -> tuple[int, int, float]:
    """
    Compute padding statistics for a batch of sequences.

    Parameters
    ----------
    lengths : list of int
        One-dimensional list containing the length of each sequence in the batch.

    Returns
    -------
    padded_tokens : int
        Total number of tokens after naïve padding (batch size times maximum length).
    packed_tokens : int
        Total number of actual tokens (sum of all lengths).
    waste_fraction : float
        Fraction of padded tokens that are wasted.
    """
    batch = len(lengths)

    max_len = 0
    if batch > 0:
        max_len = int(lengths[0])
        for i in range(1, batch):
            val = int(lengths[i])
            if val > max_len:
                max_len = val

    padded_tokens = batch * max_len

    packed_tokens = 0
    for i in range(batch):
        packed_tokens += int(lengths[i])

    waste_fraction = (padded_tokens - packed_tokens) / padded_tokens if padded_tokens > 0 else 0.0

    return padded_tokens, packed_tokens, waste_fraction

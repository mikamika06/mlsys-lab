def classify_batches(max_bucket: int, batch_sizes: list[int]) -> list[bool]:
    """
    Return a boolean list indicating whether each batch size is captured.

    Parameters
    ----------
    max_bucket : int
        Maximum size that can be captured.
    batch_sizes : list[int]
        Sizes of incoming batches.

    Returns
    -------
    list[bool]
        True where the batch size <= max_bucket, False otherwise.
    """
    res = []
    for size in batch_sizes:
        res.append(size <= max_bucket)
    return res

def classify_batches(timestamps: list[float],
                     batch_size: int,
                     timeout: float) -> list[int]:
    """
    Correct implementation of the batch‑trigger classification.
    Returns a list of labels (0=size, 1=timeout).
    """
    n = len(timestamps)
    labels = []
    i = 0
    while i < n:
        start = timestamps[i]
        count = 1
        j = i + 1
        # Add items until timeout or size reached
        while j < n and (timestamps[j] - start) < timeout and count < batch_size:
            count += 1
            j += 1
        if count == batch_size:
            labels.append(0)
        else:
            # If we ran out of data, treat as size trigger
            if j == n:
                labels.append(0)
            else:
                labels.append(1)
        i = j
    return labels

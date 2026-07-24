import numpy as np

def static_batch_steps(request_lengths: np.ndarray, batch_size: int) -> np.ndarray:
    arr = np.asarray(request_lengths, dtype=np.int64)
    n = arr.size
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n == 0:
        return np.array([0], dtype=np.int64)
    num_batches = (n + batch_size - 1) // batch_size
    padded_len = num_batches * batch_size
    pad_needed = padded_len - n
    if pad_needed > 0:
        arr_padded = np.concatenate(
            [arr, np.full(pad_needed, -np.iinfo(np.int64).max, dtype=np.int64)]
        )
    else:
        arr_padded = arr
    reshaped = arr_padded.reshape(num_batches, batch_size)
    max_per_batch = reshaped.max(axis=1)
    total_steps = int(max_per_batch.sum())
    return np.array([total_steps], dtype=np.int64)

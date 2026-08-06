import numpy as np

def static_batch_steps(request_lengths: np.ndarray, batch_size: int) -> np.ndarray:
    arr = np.asarray(request_lengths, dtype=np.int64)
    n = arr.size
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n == 0:
        return np.array([0], dtype=np.int64)
    num_batches = (n + batch_size - 1) // batch_size
    pad_val = -np.iinfo(np.int64).max
    total_steps = 0
    for i in range(num_batches):
        batch_max = pad_val
        start = i * batch_size
        for j in range(batch_size):
            idx = start + j
            if idx < n:
                val = arr[idx]
            else:
                val = pad_val
            if val > batch_max:
                batch_max = val
        total_steps += batch_max
    return np.array([total_steps], dtype=np.int64)

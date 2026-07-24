import numpy as np

def _ref(request_lengths, batch_size):
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

def grade(sol, fx) -> dict:
    cases = [
        (np.array([5, 3, 7]), 2),
        (np.array([]), 4),
        (np.arange(1, 11), 3),
        (np.array([10] * 7), 3),
        (np.random.randint(1, 20, size=15), 5)
    ]
    ok_exact = 1.0
    ok_size = 1.0
    for reqs, batch in cases:
        try:
            got = sol.static_batch_steps(reqs, batch)
            ref = _ref(reqs, batch)
        except Exception:
            return {"exact_match": 0.0, "size_ratio": 0.0}
        if not isinstance(got, np.ndarray) or got.shape != (1,):
            ok_exact = 0.0
            ok_size = 0.0
            break
        if not np.array_equal(got, ref):
            ok_exact = 0.0
        if got.nbytes != ref.nbytes:
            ok_size = 0.0
    return {"exact_match": ok_exact, "size_ratio": ok_size}

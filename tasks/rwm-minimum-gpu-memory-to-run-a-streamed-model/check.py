import numpy as np


def _oracle_min_gpu_memory(layer_bytes, K, activation_buffer):
    w = np.asarray(layer_bytes, dtype=np.int64)
    n = w.shape[0]
    k = int(min(max(K, 1), n))
    windows = np.lib.stride_tricks.sliding_window_view(w, k)
    peak = int(windows.sum(axis=1).max())
    return peak + int(activation_buffer)


def grade(sol, fx) -> dict:
    """
    Random layer-size arrays / window sizes / activation buffers; compares
    the submitted minimum GPU memory against a NumPy sliding-window oracle.
    """
    rng = np.random.default_rng(0)
    ok = 1.0
    for _ in range(8):
        n = int(rng.integers(2, 30))
        layer_bytes = rng.integers(1, 10_000, size=n).astype(np.int64)
        K = int(rng.integers(1, n + 2))  # sometimes K >= n
        activation_buffer = int(rng.integers(0, 5000))

        expected = _oracle_min_gpu_memory(layer_bytes, K, activation_buffer)
        try:
            got = sol.min_gpu_memory(layer_bytes.copy(), K, activation_buffer)
            gi = int(round(float(got)))
        except Exception:
            ok = 0.0
            break

        if gi != expected:
            ok = 0.0
            break
    return {"exact_match": ok}

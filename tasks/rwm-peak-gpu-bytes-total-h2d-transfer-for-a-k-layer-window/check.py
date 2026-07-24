import numpy as np

def grade(sol, fx) -> dict:
    def _ref(layer_sizes, window_size):
        arr = np.asarray(layer_sizes, dtype=np.int64)
        n = len(arr)
        if n == 0 or window_size <= 0:
            peak = 0
        else:
            w = min(window_size, n)
            cumsum = np.concatenate([[0], np.cumsum(arr)])
            sums = cumsum[w:] - cumsum[:-w]
            peak = int(np.max(sums)) if sums.size > 0 else int(np.sum(arr))
        total = int(np.sum(arr))
        return (peak, total)

    ok = 1.0
    for _ in range(20):
        n = np.random.randint(0, 51)          # up to 50 layers
        layer_sizes = list(np.random.randint(1, 1001, size=n))
        window_size = np.random.randint(1, max(2, n+3))  # sometimes > n
        try:
            got = sol.gpu_transfer_stats(layer_sizes, window_size)
        except Exception:
            ok = 0.0
            break
        ref = _ref(layer_sizes, window_size)
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}

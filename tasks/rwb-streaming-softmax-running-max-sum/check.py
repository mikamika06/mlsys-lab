import numpy as np


def _oracle_softmax(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    m = x64.max()
    e = np.exp(x64 - m)
    return e / e.sum()


def _cases(fx):
    rng = np.random.default_rng(4)
    cases = [(fx["scores"], 8), (fx["scores"], 5), (fx["scores"], 37)]  # 37 == whole vector in one chunk
    for _ in range(4):
        n = int(rng.integers(3, 60))
        x = rng.uniform(-30.0, 30.0, size=n).astype(np.float32)
        if n > 4:
            x[rng.integers(0, n)] = rng.choice([700.0, -700.0])
        chunk_size = int(rng.integers(1, n + 1))
        cases.append((x, chunk_size))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for scores, chunk_size in _cases(fx):
        scores = np.asarray(scores)
        ref = _oracle_softmax(scores)

        try:
            got = sol.streaming_softmax(scores.copy(), chunk_size)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}

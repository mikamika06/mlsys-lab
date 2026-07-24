import numpy as np


def _levels():
    vals = [0.0]
    for e in range(-6, 8):
        for m in range(8):
            vals.append((2.0 ** e) * (1.0 + m / 8.0))
    return np.array(sorted(set(vals)), dtype=np.float64)


_LEVELS = _levels()


def _q_e4m3(x):
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    idx = np.searchsorted(_LEVELS, ax)
    idx = np.clip(idx, 1, len(_LEVELS) - 1)
    left = _LEVELS[idx - 1]
    right = _LEVELS[idx]
    chosen = np.where((ax - left) > (right - ax), right, left)
    chosen = np.minimum(chosen, _LEVELS[-1])
    return sign * chosen


def _oracle(W):
    W = np.asarray(W, dtype=np.float64)
    result = np.empty_like(W)
    for i, row in enumerate(W):
        peak = float(np.max(np.abs(row)))
        if peak == 0:
            result[i] = row
            continue
        candidates = np.logspace(
            np.log10(peak / _LEVELS[-1]),
            np.log10(peak),
            192,
        )
        best = None
        best_loss = float("inf")
        for scale in candidates:
            restored = _q_e4m3(row / scale) * scale
            loss = float(np.sum((restored - row) ** 2))
            if loss < best_loss:
                best_loss = loss
                best = restored
        result[i] = best
    return result


def _channel_rel_err(a, b):
    num = np.linalg.norm(a - b, axis=1)
    den = np.linalg.norm(b, axis=1) + 1e-12
    return float(np.max(num / den))


def grade(sol, fx) -> dict:
    cases = [
        np.array(
            [
                [0.1, 0.3, 1.2, 8.0],
                [4.0, 4.1, 4.2, 4.3],
                [0.001, 0.01, 0.1, 1.0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [-8.0, -2.0, 0.5, 3.0],
                [12.0, 6.0, 1.0, 0.2],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ]

    worst = 0.0
    for W in cases:
        reference = _oracle(W)
        try:
            got = np.asarray(sol.fp8_channel_quantize(W), dtype=np.float64)
        except Exception:
            return {"channel_rel_err": 1e9}
        if got.shape != reference.shape or not np.all(np.isfinite(got)):
            return {"channel_rel_err": 1e9}
        worst = max(worst, _channel_rel_err(got, reference))
    return {"channel_rel_err": worst}

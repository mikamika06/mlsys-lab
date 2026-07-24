import numpy as np


def _oracle(x, block_size):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    for start in range(0, len(x), block_size):
        end = min(start + block_size, len(x))
        block = x[start:end]
        scale = np.max(np.abs(block)) / 127.0
        if scale == 0:
            out[start:end] = 0.0
            continue
        q = np.clip(np.rint(block / scale), -127, 127).astype(np.int8)
        out[start:end] = q.astype(np.float64) * scale
    return out


def _rel_err(a, b):
    return float(
        np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
        / (np.linalg.norm(np.asarray(b, dtype=np.float64)) + 1e-12)
    )


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array(
                [0.12, -0.18, 0.09, 0.15, 18.0, 0.02, -0.03, 0.04,
                 0.11, 0.08, 0.06, -0.07],
                dtype=np.float64,
            ),
            4,
        ),
        (
            np.array(
                [0.001, 0.002, -0.003, 0.004, 0.005, 9.0,
                 0.006, -0.007, 0.008, 0.009, 0.01],
                dtype=np.float64,
            ),
            3,
        ),
        (
            np.linspace(-1.0, 1.0, 257, dtype=np.float64),
            32,
        ),
    ]

    worst = 0.0
    for x, block_size in cases:
        try:
            got = sol.blockwise_quantize_dequantize(x, block_size)
        except Exception:
            return {"rel_err": float("inf")}

        ref = _oracle(x, block_size)
        worst = max(worst, _rel_err(got, ref))

    return {"rel_err": worst}

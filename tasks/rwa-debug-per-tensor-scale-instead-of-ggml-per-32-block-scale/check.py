import numpy as np


def _q4_oracle(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for start in range(0, len(x), 32):
        block = x[start:start + 32]
        scale = np.max(np.abs(block)) / 7.0
        if scale == 0:
            out[start:start + 32] = 0.0
        else:
            q = np.clip(np.round(block / scale), -8, 7)
            out[start:start + 32] = q * scale
    return out


def _tensor_scale_bug(x):
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x)) / 7.0
    return np.clip(np.round(x / scale), -8, 7) * scale


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(3):
        blocks = []
        for _ in range(8):
            magnitude = 10 ** rng.uniform(-2, 2)
            blocks.append(rng.normal(0.0, magnitude, 32))
        cases.append(np.concatenate(blocks).astype(np.float64))

    total = 0.0
    count = 0
    for x in cases:
        ref = _q4_oracle(x)
        buggy = _tensor_scale_bug(x)
        try:
            got_list = sol.q4_0_dequantize(x.tolist())
            got = np.asarray(got_list, dtype=np.float64)
        except Exception:
            return {"mse": float("inf")}

        if got.shape != x.shape:
            return {"mse": float("inf")}

        candidate_mse = float(np.mean((got - x) ** 2))
        ref_mse = float(np.mean((ref - x) ** 2))
        buggy_mse = float(np.mean((buggy - x) ** 2))

        if abs(candidate_mse - ref_mse) > 1e-10 or candidate_mse >= buggy_mse:
            return {"mse": float("inf")}

        total += candidate_mse
        count += 1

    return {"mse": total / count}

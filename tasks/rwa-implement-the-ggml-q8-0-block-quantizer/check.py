import numpy as np
from mlsys.scorers import mse

def _oracle(block):
    """Compute the reference Q8_0 quantisation for a single block."""
    x = np.asarray(block, dtype=np.float64)
    amax = np.max(np.abs(x))
    scale = 1.0 if amax == 0 else amax / 127.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, np.float16(scale)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(seed=42)
    num_blocks = 50
    blocks = rng.standard_normal(size=(num_blocks, 32)).astype(np.float32)

    ok = True
    total_mse = 0.0

    for block in blocks:
        try:
            q, s = sol.q8_0_quantize(block)
        except Exception:
            return {"mse": float("inf")}

        if not isinstance(q, np.ndarray) or q.dtype != np.int8 or q.shape != (32,):
            ok = False
            break
        if not isinstance(s, np.float16):
            ok = False
            break

        deq = q.astype(np.float32) * float(s)
        block_f64 = block.astype(np.float64)
        total_mse += mse(block_f64, deq)

    if not ok:
        return {"mse": float("inf")}

    avg_mse = total_mse / num_blocks
    return {"mse": 1.0 if avg_mse <= 1e-6 else 0.0}

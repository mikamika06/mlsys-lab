import numpy as np

def _linear_int8_mse(v):
    max_abs = np.max(np.abs(v))
    if max_abs == 0:
        return 0.0
    scale = max_abs / 127.0
    q = np.round(v / scale).clip(-128, 127)
    v_hat = q * scale
    mse = np.mean((v - v_hat) ** 2)
    return float(mse)

def _dynamic_map_mse(v):
    min_val = np.min(v)
    max_val = np.max(v)
    if min_val == max_val:
        return 0.0
    scale = (max_val - min_val) / 255.0
    q = np.round((v - min_val) / scale).clip(0, 255)
    v_hat = q * scale + min_val
    mse = np.mean((v - v_hat) ** 2)
    return float(mse)

def _oracle(v_blocks):
    winners = []
    for v in v_blocks:
        dm_mse = _dynamic_map_mse(v)
        li_mse = _linear_int8_mse(v)
        winner = 0 if dm_mse <= li_mse else 1
        winners.append(winner)
    return np.array(winners, dtype=int)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    n_blocks = 5
    v_blocks = [rng.standard_normal((4, 3)) for _ in range(n_blocks)]
    try:
        got = sol.quant_winner_per_block(v_blocks)
    except Exception:
        return {"exact_match": 0.0}
    if not isinstance(got, np.ndarray):
        return {"exact_match": 0.0}
    oracle = _oracle(v_blocks)
    ok = int(np.array_equal(got, oracle))
    return {"exact_match": float(ok)}

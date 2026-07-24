import numpy as np

def quant_winner_per_block(v_blocks):
    """
    Determine for each block whether Dynamic Map (0) or Linear Int8 (1)
    yields a lower mean‑squared error.
    """
    def _linear_int8_mse(v):
        max_abs = np.max(np.abs(v))
        if max_abs == 0:
            return 0.0
        scale = max_abs / 127.0
        q = np.round(v / scale).clip(-128, 127)
        v_hat = q * scale
        return float(np.mean((v - v_hat) ** 2))

    def _dynamic_map_mse(v):
        min_val = np.min(v)
        max_val = np.max(v)
        if min_val == max_val:
            return 0.0
        scale = (max_val - min_val) / 255.0
        q = np.round((v - min_val) / scale).clip(0, 255)
        v_hat = q * scale + min_val
        return float(np.mean((v - v_hat) ** 2))

    winners = []
    for v in v_blocks:
        dm_mse = _dynamic_map_mse(v)
        li_mse = _linear_int8_mse(v)
        winner = 0 if dm_mse <= li_mse else 1
        winners.append(winner)
    return np.array(winners, dtype=np.int32)

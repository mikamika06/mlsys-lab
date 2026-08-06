import numpy as np
import math

def quant_winner_per_block(v_blocks):
    """
    Determine for each block whether Dynamic Map (0) or Linear Int8 (1)
    yields a lower mean‑squared error.
    """
    def _linear_int8_mse(v):
        shape = v.shape
        flat = v.flat
        max_abs = 0.0
        for i in range(len(flat)):
            val = float(flat[i])
            abs_val = abs(val)
            if abs_val > max_abs:
                max_abs = abs_val
        if max_abs == 0.0:
            return 0.0
        scale = max_abs / 127.0
        
        sq_diff_sum = 0.0
        for i in range(len(flat)):
            val = float(flat[i])
            q_val = round(val / scale)
            if q_val < -128.0:
                q_val = -128.0
            elif q_val > 127.0:
                q_val = 127.0
            v_hat = q_val * scale
            diff = val - v_hat
            sq_diff_sum += diff * diff
            
        return float(sq_diff_sum / len(flat))

    def _dynamic_map_mse(v):
        shape = v.shape
        flat = v.flat
        min_val = float(flat[0])
        max_val = float(flat[0])
        for i in range(1, len(flat)):
            val = float(flat[i])
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
        if min_val == max_val:
            return 0.0
        scale = (max_val - min_val) / 255.0
        
        sq_diff_sum = 0.0
        for i in range(len(flat)):
            val = float(flat[i])
            q_val = round((val - min_val) / scale)
            if q_val < 0.0:
                q_val = 0.0
            elif q_val > 255.0:
                q_val = 255.0
            v_hat = q_val * scale + min_val
            diff = val - v_hat
            sq_diff_sum += diff * diff
            
        return float(sq_diff_sum / len(flat))

    winners = []
    for v in v_blocks:
        dm_mse = _dynamic_map_mse(v)
        li_mse = _linear_int8_mse(v)
        winner = 0 if dm_mse <= li_mse else 1
        winners.append(winner)
    return np.array(winners, dtype=np.int32)

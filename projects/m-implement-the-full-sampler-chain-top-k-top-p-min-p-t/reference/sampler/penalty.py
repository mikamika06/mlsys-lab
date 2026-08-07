import numpy as np


def apply_repeat_penalty(logits, input_ids, penalty=1.0, repeat_last_n=64):
    out = np.array(logits, dtype=np.float64, copy=True)
    if penalty == 1.0 or input_ids is None or len(input_ids) == 0 or repeat_last_n <= 0:
        return out
    window = input_ids[-repeat_last_n:]
    unique_ids = np.unique(window)
    for token_id in unique_ids:
        if 0 <= token_id < len(out):
            val = out[token_id]
            if val < 0:
                out[token_id] = val * penalty
            else:
                out[token_id] = val / penalty
    return out

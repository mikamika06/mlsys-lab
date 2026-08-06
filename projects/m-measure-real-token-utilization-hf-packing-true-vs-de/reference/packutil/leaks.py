import numpy as np

def detect_attention_leak(position_ids, attention_mask):
    pos = np.array(position_ids)
    mask = np.array(attention_mask)
    diffs = np.diff(pos)
    reset_indices = np.where(diffs < 0)[0] + 1

    leaks_found = 0
    for idx in reset_indices:
        if idx < mask.shape[0]:
            row = mask[idx]
            for prev_idx in range(idx):
                if row[prev_idx] == 1:
                    leaks_found += 1
    return int(leaks_found > 0)

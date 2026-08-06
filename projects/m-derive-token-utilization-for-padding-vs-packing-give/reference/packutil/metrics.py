import numpy as np

def compute_utilization(lengths, max_length):
    lengths = np.array(lengths)
    total_actual = np.sum(lengths)
    total_padding_slots = len(lengths) * max_length
    padding_util = total_actual / total_padding_slots if total_padding_slots > 0 else 0.0
    current_bin = 0
    bins_used = 0
    for l in sorted(lengths, reverse=True):
        if l > max_length:
            bins_used += int(np.ceil(l / max_length))
        else:
            if current_bin + l > max_length:
                bins_used += 1
                current_bin = l
            else:
                current_bin += l
    if current_bin > 0:
        bins_used += 1
    packing_util = total_actual / (bins_used * max_length) if bins_used > 0 else 0.0
    return {"padding_utilization": float(padding_util), "packing_utilization": float(packing_util)}

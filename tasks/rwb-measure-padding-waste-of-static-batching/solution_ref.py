import numpy as np


def padding_waste_fraction(lens: np.ndarray, batch_ids: np.ndarray) -> float:
    """
    lens      : 1-D int array, total (prompt + generation) token length of
                each request.
    batch_ids : 1-D int array, same length as `lens`; batch_ids[i] is the
                STATIC batch request i was assigned to (static batching pads
                every request in a batch out to that batch's longest member).

    For each batch b with requests of lengths L_b = {len_i : batch_ids[i]=b},
    the number of token SLOTS actually allocated is
    max(L_b) * |L_b|, of which max(L_b)*|L_b| - sum(L_b) are pure padding
    waste. Return the total wasted fraction across all batches:

        wasted_fraction = sum_b [max(L_b)*|L_b| - sum(L_b)] / sum_b [max(L_b)*|L_b|]
    """
    lens = np.asarray(lens, dtype=np.float64)
    batch_ids = np.asarray(batch_ids)

    total_slots = 0.0
    total_wasted = 0.0
    for b in np.unique(batch_ids):
        batch_lens = lens[batch_ids == b]
        max_len = float(np.max(batch_lens))
        batch_size = float(batch_lens.shape[0])
        slots = max_len * batch_size
        total_slots += slots
        total_wasted += slots - float(np.sum(batch_lens))

    return total_wasted / total_slots

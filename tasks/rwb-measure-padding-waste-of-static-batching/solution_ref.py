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

    unique_batches = []
    i = 0
    while i < batch_ids.shape[0]:
        val = batch_ids[i]
        found = False
        j = 0
        while j < len(unique_batches):
            if unique_batches[j] == val:
                found = True
                break
            j += 1
        if not found:
            unique_batches.append(val)
        i += 1

    total_slots = 0.0
    total_wasted = 0.0
    
    b_idx = 0
    while b_idx < len(unique_batches):
        b = unique_batches[b_idx]
        
        batch_lens_list = []
        k = 0
        while k < batch_ids.shape[0]:
            if batch_ids[k] == b:
                batch_lens_list.append(lens[k])
            k += 1

        max_len = 0.0
        if len(batch_lens_list) > 0:
            max_len = batch_lens_list[0]
            m = 1
            while m < len(batch_lens_list):
                if batch_lens_list[m] > max_len:
                    max_len = batch_lens_list[m]
                m += 1

        batch_size = float(len(batch_lens_list))
        
        sum_lens = 0.0
        m = 0
        while m < len(batch_lens_list):
            sum_lens += batch_lens_list[m]
            m += 1

        slots = max_len * batch_size
        total_slots += slots
        total_wasted += slots - sum_lens

        b_idx += 1

    return total_wasted / total_slots

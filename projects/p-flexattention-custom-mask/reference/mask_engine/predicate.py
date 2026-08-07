import numpy as np

def make_mask_predicate(doc_ids, window_size):
    doc_arr = np.asarray(doc_ids)
    def predicate(q_idx, kv_idx):
        same_doc = (doc_arr[q_idx] == doc_arr[kv_idx])
        in_window = (q_idx - kv_idx >= 0) & (q_idx - kv_idx <= window_size)
        return same_doc & in_window
    return predicate

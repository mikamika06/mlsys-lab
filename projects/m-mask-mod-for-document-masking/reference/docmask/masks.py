def document_mask_mod(doc_ids):
    def mask_mod(b, h, q_idx, kv_idx):
        return doc_ids[b, q_idx] == doc_ids[b, kv_idx]
    return mask_mod

def prefix_lm_mask_mod(prefix_lengths):
    def mask_mod(b, h, q_idx, kv_idx):
        p_len = prefix_lengths[b]
        is_prefix_kv = kv_idx < p_len
        is_causal = q_idx >= kv_idx
        return is_prefix_kv | is_causal
    return mask_mod

import numpy as np

def ref_document_mask_mod(doc_ids):
    def mask_mod(b, h, q_idx, kv_idx):
        return bool(doc_ids[b, q_idx] == doc_ids[b, kv_idx])
    return mask_mod

def ref_prefix_lm_mask_mod(prefix_lengths):
    def mask_mod(b, h, q_idx, kv_idx):
        p_len = prefix_lengths[b]
        return bool(kv_idx < p_len or q_idx >= kv_idx)
    return mask_mod

def ref_block_sparsity_fraction(mask_fn, seq_len, block_size=128):
    total_blocks = (seq_len + block_size - 1) // block_size
    masked_blocks = 0
    for i in range(total_blocks):
        for j in range(total_blocks):
            q_start = i * block_size
            q_end = min((i + 1) * block_size, seq_len)
            kv_start = j * block_size
            kv_end = min((j + 1) * block_size, seq_len)
            any_allowed = False
            for q in range(q_start, q_end):
                for kv in range(kv_start, kv_end):
                    if mask_fn(0, 0, q, kv):
                        any_allowed = True
                        break
                if any_allowed:
                    break
            if not any_allowed:
                masked_blocks += 1
    return float(masked_blocks / (total_blocks * total_blocks))

TEST_CASES_DOC = [
    np.array([[0, 0, 1, 1, 1]]),
    np.array([[0, 1, 2, 2]])
]

TEST_CASES_PREFIX = [
    [2],
    [4]
]

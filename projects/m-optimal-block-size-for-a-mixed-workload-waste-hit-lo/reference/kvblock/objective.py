import numpy as np


def calculate_internal_fragmentation(seq_lengths, block_size):
    lengths = np.asarray(seq_lengths, dtype=np.int64)
    remainder = lengths % block_size
    waste = np.where(remainder == 0, 0, block_size - remainder)
    return int(np.sum(waste))


def calculate_prefix_truncation_loss(shared_prefixes, request_prefixes, block_size):
    total_loss = 0
    for req_p in request_prefixes:
        req_len = len(req_p)
        best_matched_tokens = 0
        best_cached_full_tokens = 0
        for sh_p in shared_prefixes:
            common = 0
            min_l = min(req_len, len(sh_p))
            while common < min_l and req_p[common] == sh_p[common]:
                common += 1
            full_block_tokens = (common // block_size) * block_size
            if common > best_matched_tokens or (common == best_matched_tokens and full_block_tokens > best_cached_full_tokens):
                best_matched_tokens = common
                best_cached_full_tokens = full_block_tokens
        loss = best_matched_tokens - best_cached_full_tokens
        total_loss += loss
    return total_loss


def evaluate_workload_objective(seq_lengths, shared_prefixes, request_prefixes, block_size, hit_penalty_weight):
    frag_waste = calculate_internal_fragmentation(seq_lengths, block_size)
    trunc_loss = calculate_prefix_truncation_loss(shared_prefixes, request_prefixes, block_size)
    return float(frag_waste + hit_penalty_weight * trunc_loss)

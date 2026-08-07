import numpy as np

SAMPLE_SEQS = [17, 32, 65, 120, 4, 511, 1024, 1025]
CANDIDATE_BLOCK_SIZES = [4, 8, 16, 32, 64]

SHARED_PREFIXES = [
    tuple(range(1, 33)),
    tuple(range(1, 65)),
    tuple(range(100, 150)),
]

REQUEST_PREFIXES = [
    tuple(range(1, 20)),
    tuple(range(1, 40)),
    tuple(range(1, 65)),
    tuple(range(100, 130)),
    tuple(range(200, 220)),
]

TRACE = [
    {"seq_len": 100, "prefix": tuple(range(1, 35))},
    {"seq_len": 250, "prefix": tuple(range(1, 60))},
    {"seq_len": 45, "prefix": tuple(range(100, 125))},
    {"seq_len": 512, "prefix": tuple(range(1, 65))},
    {"seq_len": 12, "prefix": tuple(range(500, 510))},
]

TOTAL_MEMORY_BLOCKS = {
    4: 1000,
    8: 500,
    16: 250,
    32: 100,
    64: 50,
}

HIT_PENALTY_WEIGHT = 4.0


def ref_calculate_internal_fragmentation(seq_lengths, block_size):
    lengths = np.asarray(seq_lengths, dtype=np.int64)
    remainder = lengths % block_size
    waste = np.where(remainder == 0, 0, block_size - remainder)
    return int(np.sum(waste))


def ref_calculate_prefix_truncation_loss(shared_prefixes, request_prefixes, block_size):
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


def ref_evaluate_workload_objective(seq_lengths, shared_prefixes, request_prefixes, block_size, hit_penalty_weight):
    frag = ref_calculate_internal_fragmentation(seq_lengths, block_size)
    trunc = ref_calculate_prefix_truncation_loss(shared_prefixes, request_prefixes, block_size)
    return float(frag + hit_penalty_weight * trunc)


def ref_simulate_block_sweep(trace, candidate_block_sizes, total_memory_blocks, hit_penalty_weight):
    results = {}
    for b_size in candidate_block_sizes:
        seq_lengths = [req["seq_len"] for req in trace]
        shared_prefixes = list({tuple(req["prefix"]) for req in trace if "prefix" in req})
        request_prefixes = [tuple(req["prefix"]) for req in trace if "prefix" in req]

        base_obj = ref_evaluate_workload_objective(
            seq_lengths, shared_prefixes, request_prefixes, b_size, hit_penalty_weight
        )

        total_allocated_blocks = sum((l + b_size - 1) // b_size for l in seq_lengths)
        capacity = total_memory_blocks.get(b_size, 1000000)
        overflow_penalty = max(0, total_allocated_blocks - capacity) * b_size * 2.0

        results[b_size] = float(base_obj + overflow_penalty)
    return results


def ref_find_optimal_block_size(trace, candidate_block_sizes, total_memory_blocks, hit_penalty_weight):
    costs = ref_simulate_block_sweep(trace, candidate_block_sizes, total_memory_blocks, hit_penalty_weight)
    sorted_sizes = sorted(candidate_block_sizes)
    return min(sorted_sizes, key=lambda b: (costs[b], b))

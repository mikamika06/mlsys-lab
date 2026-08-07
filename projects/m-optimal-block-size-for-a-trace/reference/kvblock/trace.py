"""Trace analysis and block size optimization."""


def total_overhead(trace, candidate_block_sizes, bytes_per_tok=128, metadata_bytes_per_block=64):
    """Calculate total overhead bytes for each candidate block size."""
    overheads = []
    for b in candidate_block_sizes:
        total = 0
        for seq_len in trace:
            num_blocks = (seq_len + b - 1) // b
            frag_tokens = (b - (seq_len % b)) % b
            frag_bytes = frag_tokens * bytes_per_tok
            meta_bytes = num_blocks * metadata_bytes_per_block
            total += frag_bytes + meta_bytes
        overheads.append(total)
    return overheads


def find_optimal_block_size(trace, candidate_block_sizes, bytes_per_tok=128, metadata_bytes_per_block=64):
    """Find the block size from candidates that minimizes total overhead."""
    costs = total_overhead(trace, candidate_block_sizes, bytes_per_tok, metadata_bytes_per_block)
    best_idx = 0
    min_cost = costs[0]
    for i in range(1, len(costs)):
        if costs[i] < min_cost:
            min_cost = costs[i]
            best_idx = i
    return candidate_block_sizes[best_idx]

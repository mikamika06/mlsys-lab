"""Reference data and oracles for test harness."""

import random

random.seed(42)

CANDIDATES = [8, 16, 32, 64, 128]

TRACES = [
    [random.randint(16, 4096) for _ in range(50)],
    [random.randint(128, 2048) for _ in range(100)],
    [random.randint(1, 512) for _ in range(200)],
    [random.randint(1024, 8192) for _ in range(30)],
]


def total_overhead(trace, candidate_block_sizes, bytes_per_tok=128, metadata_bytes_per_block=64):
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
    costs = total_overhead(trace, candidate_block_sizes, bytes_per_tok, metadata_bytes_per_block)
    best_idx = 0
    min_cost = costs[0]
    for i in range(1, len(costs)):
        if costs[i] < min_cost:
            min_cost = costs[i]
            best_idx = i
    return candidate_block_sizes[best_idx]


def simulate_prefix_cache(requests, block_size, max_blocks):
    cache = {}
    usage = {}
    clock = 0
    next_block_id = 1
    total_hits = 0
    total_blocks = 0

    for token_ids in requests:
        num_blocks = len(token_ids) // block_size
        if num_blocks == 0:
            continue
        blocks = [tuple(token_ids[i * block_size : (i + 1) * block_size]) for i in range(num_blocks)]
        hits = 0
        matched = True
        for i, blk in enumerate(blocks):
            prefix_key = tuple(blocks[: i + 1])
            clock += 1
            if matched and prefix_key in cache:
                hits += 1
                usage[prefix_key] = clock
            else:
                matched = False
                if len(cache) >= max_blocks and prefix_key not in cache:
                    lru = min(usage.keys(), key=lambda k: usage[k])
                    del cache[lru]
                    del usage[lru]
                if prefix_key not in cache:
                    cache[prefix_key] = next_block_id
                    next_block_id += 1
                usage[prefix_key] = clock
        total_hits += hits
        total_blocks += num_blocks

    rate = total_hits / total_blocks if total_blocks > 0 else 0.0
    return total_hits, total_blocks, rate


def reference_triage(block_table, total_seq_len, block_size, max_valid_block_id):
    expected_blocks = (total_seq_len + block_size - 1) // block_size if total_seq_len > 0 else 0
    repaired = list(block_table[:expected_blocks])
    issues = []

    if len(block_table) != expected_blocks:
        issues.append("length_mismatch")
        while len(repaired) < expected_blocks:
            repaired.append(-1)

    seen = set()
    for i in range(len(repaired)):
        blk_id = repaired[i]
        if blk_id < 0 or blk_id > max_valid_block_id or blk_id in seen:
            issues.append(f"invalid_or_duplicate_block_{i}")
            repaired[i] = -1
        else:
            seen.add(blk_id)

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "expected_blocks": expected_blocks,
        "repaired_table": repaired,
    }

import math


def calculate_paged_kv_plan(seq_lens, block_size, page_budget):
    total_blocks_needed = 0
    blocks_per_seq = []
    total_tokens = sum(seq_lens)

    for length in seq_lens:
        if length == 0:
            blocks_per_seq.append(0)
            continue
        needed = math.ceil(length / block_size)
        blocks_per_seq.append(needed)
        total_blocks_needed += needed

    allocated_blocks = min(total_blocks_needed, page_budget)
    total_capacity_tokens = allocated_blocks * block_size
    waste_tokens = total_capacity_tokens - total_tokens if total_blocks_needed <= page_budget else (allocated_blocks * block_size - sum(min(l, allocated_blocks * block_size) for l in seq_lens))
    waste_tokens = max(0, waste_tokens)

    efficiency = total_tokens / total_capacity_tokens if total_capacity_tokens > 0 else 0.0

    return {
        "total_blocks_needed": total_blocks_needed,
        "allocated_blocks": allocated_blocks,
        "blocks_per_seq": blocks_per_seq,
        "waste_tokens": waste_tokens,
        "efficiency": efficiency,
        "fits_in_budget": total_blocks_needed <= page_budget
    }


def simulate_block_allocation(request_arrival_pattern, block_size, page_budget):
    active_blocks = 0
    peak_blocks = 0
    completed = 0
    dropped = 0

    for req in request_arrival_pattern:
        action = req.get("action", "arrive")
        seq_len = req.get("seq_len", 0)
        needed = math.ceil(seq_len / block_size) if seq_len > 0 else 0

        if action == "arrive":
            if active_blocks + needed <= page_budget:
                active_blocks += needed
                peak_blocks = max(peak_blocks, active_blocks)
                completed += 1
            else:
                dropped += 1
        elif action == "depart":
            active_blocks = max(0, active_blocks - needed)

    return {
        "peak_blocks": peak_blocks,
        "active_blocks": active_blocks,
        "completed": completed,
        "dropped": dropped,
        "utilization": peak_blocks / page_budget if page_budget > 0 else 0.0
    }

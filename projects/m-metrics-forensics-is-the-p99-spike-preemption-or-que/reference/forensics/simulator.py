def simulate_server(requests, block_manager_capacity, max_num_seqs):
    active_seqs = 0
    preemptions = 0
    total_queue_delay = 0.0
    free_blocks = block_manager_capacity

    for req in requests:
        tokens = req["tokens"]
        blocks_needed = (tokens + 15) // 16
        while active_seqs >= max_num_seqs or free_blocks < blocks_needed:
            if active_seqs > 0 and free_blocks < blocks_needed:
                preemptions += 1
                free_blocks += 4
                active_seqs -= 1
            else:
                total_queue_delay += 0.05
                break
        active_seqs += 1
        free_blocks -= blocks_needed

    return {
        "preemption_count": preemptions,
        "total_queue_delay": round(total_queue_delay, 4)
    }

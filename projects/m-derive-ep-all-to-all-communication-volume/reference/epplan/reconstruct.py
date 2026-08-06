def reconstruct_token_counts(dispatch_log: list[dict], num_ranks: int) -> list[int]:
    counts = [0] * num_ranks
    for entry in dispatch_log:
        dst = entry.get("dst_rank")
        count = entry.get("token_count", 0)
        if dst is not None and 0 <= dst < num_ranks:
            counts[dst] += count
    return counts

def compute_split_count(batch_size, num_heads, kv_len, block_size, num_sms, target_waves=1, max_splits=128):
    if kv_len <= 0 or block_size <= 0:
        return 1
    num_tiles = (kv_len + block_size - 1) // block_size
    if num_tiles <= 1:
        return 1
    total_heads = batch_size * num_heads
    if total_heads <= 0:
        return 1
    target_tasks = num_sms * target_waves
    desired_splits = (target_tasks + total_heads - 1) // total_heads
    splits = max(1, min(desired_splits, num_tiles, max_splits))
    return int(splits)


def partition_kv_ranges(kv_len, split_count):
    if split_count <= 1 or kv_len <= 1:
        return [(0, kv_len)]
    split_count = min(split_count, kv_len)
    base = kv_len // split_count
    rem = kv_len % split_count
    ranges = []
    curr = 0
    for i in range(split_count):
        length = base + (1 if i < rem else 0)
        ranges.append((curr, curr + length))
        curr += length
    return ranges

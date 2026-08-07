def compute_split_count(batch_size, num_heads, kv_len, block_size, num_sms, target_waves=1, max_splits=128):
    raise NotImplementedError


def partition_kv_ranges(kv_len, split_count):
    raise NotImplementedError

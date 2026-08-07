def optimize_buckets(tensors, max_size):
    buckets = []
    current_bucket = []
    current_size = 0
    for name, size in tensors:
        if current_size + size > max_size and current_bucket:
            buckets.append(current_bucket)
            current_bucket = [(name, size)]
            current_size = size
        else:
            current_bucket.append((name, size))
            current_size += size
    if current_bucket:
        buckets.append(current_bucket)
    return buckets

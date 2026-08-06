def assign_buckets(parameters, bucket_cap_mb):
    buckets = []
    current_bucket = []
    current_size = 0
    cap_bytes = bucket_cap_mb * 1024 * 1024

    for p in reversed(parameters):
        p_size = p.get("size_bytes", 4)
        if current_size + p_size > cap_bytes and current_bucket:
            buckets.append(current_bucket)
            current_bucket = [p["name"]]
            current_size = p_size
        else:
            current_bucket.append(p["name"])
            current_size += p_size
    if current_bucket:
        buckets.append(current_bucket)
    return buckets

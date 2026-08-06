def build_bucket_plan(params, bucket_cap_bytes):
    buckets = []
    current_bucket = []
    current_bytes = 0
    for p in reversed(params):
        p_bytes = p["numel"] * p["element_size"]
        if current_bucket and (current_bytes + p_bytes > bucket_cap_bytes):
            buckets.append(current_bucket)
            current_bucket = [p["name"]]
            current_bytes = p_bytes
        else:
            current_bucket.append(p["name"])
            current_bytes += p_bytes
    if current_bucket:
        buckets.append(current_bucket)
    return buckets

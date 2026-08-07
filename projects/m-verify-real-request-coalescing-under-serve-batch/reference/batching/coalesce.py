def verify_coalescing(requests, max_batch_size, batch_wait_timeout_s):
    batches = []
    current_batch = []
    for req in requests:
        current_batch.append(req)
        if len(current_batch) >= max_batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)
    return batches

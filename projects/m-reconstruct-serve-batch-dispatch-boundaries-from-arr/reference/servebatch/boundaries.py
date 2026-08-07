def reconstruct_boundaries(arrivals, max_batch_size, timeout_s):
    if not arrivals:
        return []
    sorted_arr = sorted(arrivals)
    batches = []
    curr = [sorted_arr[0]]
    start_time = sorted_arr[0]
    for t in sorted_arr[1:]:
        if len(curr) >= max_batch_size or (t - start_time) > timeout_s:
            batches.append(curr)
            curr = [t]
            start_time = t
        else:
            curr.append(t)
    if curr:
        batches.append(curr)
    return batches

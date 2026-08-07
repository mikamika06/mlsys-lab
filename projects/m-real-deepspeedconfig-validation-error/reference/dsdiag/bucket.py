def optimal_bucket_size(tensor_sizes, memory_ceiling):
    best_size = 0
    for size in sorted(tensor_sizes):
        if size <= memory_ceiling:
            best_size = size
    if best_size == 0 and tensor_sizes:
        best_size = min(tensor_sizes)
    return best_size

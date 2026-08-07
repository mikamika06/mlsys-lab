def calculate_stats(shards):
    total_bytes = 0
    total_params = 0
    for s in shards:
        for w in s.tensors.values():
            total_bytes += w.nbytes
            total_params += w.size
    bpw = (total_bytes * 8) / total_params if total_params > 0 else 0
    return {
        "total_bytes": total_bytes,
        "total_params": total_params,
        "bpw": bpw
    }

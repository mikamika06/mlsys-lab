from partitioner.predict import partition_ops

def optimize_allowlist(ops, base_allowlist, candidates, op_sizes, blob_overhead):
    def cost(allowlist):
        parts = partition_ops(ops, allowlist)
        blobs = set(p for p in parts if p != -1)
        num_blobs = len(blobs)
        byte_sum = sum(op_sizes.get(op['type'], 0) for i, op in enumerate(ops) if parts[i] != -1)
        return byte_sum + num_blobs * blob_overhead

    best_cand = None
    best_cost = cost(base_allowlist)
    for cand in candidates:
        cand_al = set(base_allowlist) | {cand}
        c = cost(cand_al)
        if c < best_cost:
            best_cost = c
            best_cand = cand
    return best_cand, float(best_cost)

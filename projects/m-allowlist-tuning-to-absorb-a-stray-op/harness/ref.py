import numpy as np

def generate_fixtures():
    rng = np.random.RandomState(42)
    types = ['Conv2D', 'Relu', 'Add', 'Mul', 'Cast', 'Reshape', 'Transpose', 'MaxPool', 'AvgPool', 'Concat']
    configs = []
    for _ in range(20):
        ops = [{'id': i, 'type': rng.choice(types)} for i in range(12)]
        base_allowlist = set(rng.choice(types, size=3, replace=False))
        candidates = set(rng.choice([t for t in types if t not in base_allowlist], size=3, replace=False))
        op_sizes = {t: int(rng.randint(100, 1000)) for t in types}
        blob_overhead = int(rng.randint(1500, 4000))
        configs.append({
            'ops': ops,
            'base_allowlist': base_allowlist,
            'candidates': candidates,
            'op_sizes': op_sizes,
            'blob_overhead': blob_overhead
        })
    return configs

CONFIGS = generate_fixtures()

def partition_ops(ops, allowlist):
    res = []
    current_blob = -1
    in_blob = False
    for op in ops:
        if op['type'] in allowlist:
            if not in_blob:
                current_blob += 1
                in_blob = True
            res.append(current_blob)
        else:
            in_blob = False
            res.append(-1)
    return res

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

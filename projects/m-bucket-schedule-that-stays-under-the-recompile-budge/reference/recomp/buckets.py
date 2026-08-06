def compute_buckets(shapes, budget):
    if not shapes:
        return []
    unique_shapes = sorted(list(set(shapes)))
    if len(unique_shapes) <= budget:
        return unique_shapes
    step = len(unique_shapes) / float(budget)
    buckets = []
    for i in range(budget):
        idx = int(i * step)
        buckets.append(unique_shapes[idx])
    return sorted(list(set(buckets)))

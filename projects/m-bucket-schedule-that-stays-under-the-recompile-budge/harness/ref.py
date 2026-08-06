import random

def get_fixtures():
    rng = random.Random(42)
    shapes = [rng.randint(1, 1024) for _ in range(50)]
    shapes.sort()
    budget = 5
    return shapes, budget

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

def minimal_specializations(traces, budget):
    if not traces:
        return []
    unique = sorted(list(set(traces)))
    if len(unique) <= budget:
        return unique
    step = (len(unique) - 1) / float(budget - 1) if budget > 1 else 1
    specs = [unique[int(round(i * step))] for i in range(budget)]
    return sorted(list(set(specs)))

def validate_guard(shape, bucket, guard_type):
    if guard_type == "shape":
        return shape <= bucket
    elif guard_type == "value":
        return shape == bucket
    elif guard_type == "id":
        return shape == bucket
    return False

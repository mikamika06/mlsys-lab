import random


def get_test_cases():
    rng = random.Random(42)
    cases = []
    for _ in range(10):
        n = rng.randint(4, 16)
        max_len = 512
        lengths = [rng.randint(32, max_len) for _ in range(n)]
        cases.append((lengths, max_len))
    return cases


def compute_waste_percentage(lengths, max_length):
    if not lengths:
        return 0.0
    total_capacity = len(lengths) * max_length
    total_actual = sum(lengths)
    return ((total_capacity - total_actual) / total_capacity) * 100.0


def compute_attention_costs(lengths, max_length):
    padding_cost = len(lengths) * (max_length ** 2)
    packing_cost = sum(l ** 2 for l in lengths)
    return {"padding_cost": float(padding_cost), "packing_cost": float(packing_cost)}


def compute_batch_counts(lengths, max_length):
    padding_batches = len(lengths)
    current = 0
    packing_batches = 1
    for l in sorted(lengths, reverse=True):
        if current + l <= max_length:
            current += l
        else:
            packing_batches += 1
            current = l
    return {"padding_batches": padding_batches, "packing_batches": packing_batches}

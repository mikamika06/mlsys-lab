import random


def generate_cases():
    rng = random.Random(42)
    cases = []
    for _ in range(10):
        n_nodes = rng.randint(10, 30)
        nodes = []
        for i in range(n_nodes):
            is_ew = rng.choice([True, False])
            b = rng.randint(64, 4096) * 1024
            nodes.append({"id": i, "is_elementwise": is_ew, "bytes": b})
        n_breaks = rng.randint(1, 4)
        breaks = sorted(rng.sample(range(1, n_nodes), min(n_breaks, n_nodes - 1)))
        cases.append((nodes, breaks))
    return cases

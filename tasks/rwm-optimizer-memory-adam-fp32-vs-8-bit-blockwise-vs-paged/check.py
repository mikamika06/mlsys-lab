import random

def grade(sol, fx) -> dict:
    # Generate a handful of random test cases
    cases = []
    for _ in range(5):
        num_params = random.randint(1_000, 10_000_000)
        block_size = random.choice([32, 64, 128, 256, 512, 1024])
        cases.append((num_params, block_size))

    ok = 1.0
    for n, b in cases:
        try:
            got = sol.estimate_memory(n, b)
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(got, (tuple, list)) or len(got) != 3:
            return {"exact_match": 0.0}

        fp32_ref = 8 * n
        blocks = (n + b - 1) // b
        blockwise_ref = 2 * n + 2 * blocks * 4
        paged_ref = 2 * n

        if tuple(map(int, got)) != (fp32_ref, blockwise_ref, paged_ref):
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}

import math
import random

def _oracle(lengths, block_size, total_blocks):
    """Greedy shortest-first oracle: sort by ceil(len/bs), admit until exhausted."""
    blocks = sorted((l + block_size - 1) // block_size for l in lengths)
    used = 0
    count = 0
    for b in blocks:
        if used + b <= total_blocks:
            used += b
            count += 1
        else:
            break
    return count

def grade(sol, fx) -> dict:
    # Hand-picked edge cases
    cases = [
        ([], 8, 100),                 # empty list → 0
        ([0, 0, 0], 8, 10),           # zero-length sequences → all fit
        ([5], 8, 0),                  # pool empty, sequence needs 1 → 0
        ([5], 8, 1),                  # pool=1, sequence needs 1 → 1
        ([10, 10, 10], 10, 2),        # exact multiples, pool limits
        ([8, 8, 8, 8], 8, 3),         # four seqs of exactly 1 block, pool 3
        ([7, 7, 7, 7], 8, 3),         # non-multiple, ceil=1 each, pool 3
        ([1, 2, 3, 4, 5], 3, 5),      # mixed, needs [1,1,1,2,2]
        ([1, 1, 1, 1, 1], 1, 5),      # each needs exactly 1 block
        ([100, 200, 50], 10, 20),     # varying sizes
        ([9, 9, 9], 10, 2),           # each needs 1 block, pool 2
        ([11, 11, 11], 10, 2),        # each needs 2 blocks, pool 2 → 1
    ]

    # Deterministic pseudo-random cases (seed 42 for reproducibility)
    rng = random.Random(42)
    for _ in range(20):
        n = rng.randint(1, 60)
        lengths = [rng.randint(0, 500) for _ in range(n)]
        block_size = rng.randint(1, 100)
        total_blocks = rng.randint(0, 1000)
        cases.append((lengths, block_size, total_blocks))

    ok = 1.0
    for lengths, block_size, total_blocks in cases:
        try:
            got = sol.max_concurrent_sequences(list(lengths), block_size, total_blocks)
        except Exception:
            ok = 0.0
            break
        expected = _oracle(lengths, block_size, total_blocks)
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}

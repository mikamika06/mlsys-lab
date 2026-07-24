import math
import random

def _oracle(seq_len, block_size, num_layers, num_kv_heads, head_dim, dtype_bytes):
    """Reference computation of preemption costs (must NOT be hard-coded)."""
    num_blocks = math.ceil(seq_len / block_size)
    per_token = num_layers * num_kv_heads * head_dim * dtype_bytes
    swap_cost = 2 * num_blocks * block_size * per_token
    recompute_cost = seq_len
    return (swap_cost, recompute_cost)

def grade(sol, fx) -> dict:
    rng = random.Random(42)

    cases = [
        # Fixed: aligned length
        (16, 16, 32, 8, 128, 2),
        # Fixed: non-aligned, off by one
        (17, 16, 32, 8, 128, 2),
        # Fixed: single token
        (1, 16, 32, 8, 128, 2),
        # Fixed: large sequence, small blocks
        (100, 16, 32, 8, 128, 2),
        # Fixed: different block sizes
        (1000, 64, 24, 4, 64, 4),
        (255, 128, 80, 16, 256, 2),
    ]

    # Deterministic random cases
    for _ in range(14):
        seq_len = rng.randint(1, 4096)
        block_size = rng.choice([8, 16, 32, 64, 128])
        num_layers = rng.choice([16, 24, 32, 48, 80])
        num_kv_heads = rng.choice([1, 2, 4, 8, 16])
        head_dim = rng.choice([64, 80, 128, 256])
        dtype_bytes = rng.choice([2, 4])
        cases.append((seq_len, block_size, num_layers, num_kv_heads, head_dim, dtype_bytes))

    ok = 1.0
    for params in cases:
        expected = _oracle(*params)
        try:
            got = sol.preemption_costs(*params)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}

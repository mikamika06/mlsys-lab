import random


def prefill_cost(tokens: int, cached_prefix: int, c_attn: float, c_mlp: float) -> float:
    attn_pairs = (tokens * cached_prefix) + (tokens * (tokens + 1) // 2)
    return float(c_attn * attn_pairs + c_mlp * tokens)


def simulate_batch(doc_tokens: int, question_lengths: list[int], c_attn: float, c_mlp: float) -> tuple[float, float]:
    baseline = 0.0
    for q in question_lengths:
        baseline += prefill_cost(doc_tokens + q, 0, c_attn, c_mlp)

    cached = prefill_cost(doc_tokens, 0, c_attn, c_mlp)
    for q in question_lengths:
        cached += prefill_cost(q, doc_tokens, c_attn, c_mlp)

    return baseline, cached


random.seed(42)

M1_CASES = [
    (100000, 0, 1e-8, 1e-6),
    (50, 100000, 1e-8, 1e-6),
    (128, 512, 1.5e-8, 2.0e-6),
    (2048, 8192, 1e-8, 1e-6)
]
for _ in range(20):
    t = random.randint(10, 1000)
    p = random.randint(0, 100000)
    M1_CASES.append((t, p, 1e-8, 1e-6))

M2_CASES = [
    (100000, [50]*50, 1e-8, 1e-6),
    (50000, [random.randint(20, 100) for _ in range(50)], 1e-8, 1e-6),
    (10000, [random.randint(10, 30) for _ in range(10)], 1e-8, 1e-6),
    (8192, [128, 64, 256, 32], 1.5e-8, 2.0e-6)
]

import random

def grade(sol, fx) -> dict:
    # Generate deterministic but varied test cases
    rng = random.Random(0)
    ok = 1.0
    for _ in range(10):
        K = rng.randint(0, 10000)
        phi = rng.choice([2, 4])  # typical float16 or float32 sizes
        offload_gradients = rng.choice([True, False])
        try:
            got = sol.gpu_bytes_freed(K, phi, offload_gradients)
        except Exception:
            return {"exact_match": 0.0}
        expected = K * phi * (1 + 2 * int(offload_gradients))
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}

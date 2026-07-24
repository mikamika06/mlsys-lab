import math


def _oracle(depth, seq, hidden, dtype_bytes):
    activation = seq * hidden * dtype_bytes
    return {
        "full_store": depth * activation,
        "checkpoint": math.ceil(math.sqrt(depth)) * activation,
        "offload": 0,
    }


def grade(sol, fx) -> dict:
    cases = [
        (1, 128, 256, 2),
        (12, 512, 768, 2),
        (24, 1024, 1024, 2),
        (48, 2048, 4096, 2),
        (96, 512, 1536, 4),
        (175, 256, 8192, 2),
    ]

    ok = 1.0
    for depth, seq, hidden, dtype_bytes in cases:
        try:
            got = sol.activation_memory(depth, seq, hidden, dtype_bytes)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(depth, seq, hidden, dtype_bytes):
            ok = 0.0
            break

    return {"exact_match": ok}

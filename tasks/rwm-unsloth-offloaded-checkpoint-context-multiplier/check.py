import numpy as np


def _oracle(batch_size, hidden_size, element_bytes, gpu_budget_bytes, sequence_length):
    b = np.int64(batch_size)
    h = np.int64(hidden_size)
    s = np.int64(element_bytes)
    g = np.float64(gpu_budget_bytes)

    standard_bytes_per_token = b * h * s
    offloaded_bytes = b * h * s

    standard_max_context = g / np.float64(standard_bytes_per_token)
    offloaded_max_context = g / np.float64(offloaded_bytes)

    # Standard checkpoint stores sequence_length boundary states.
    standard_gpu_bytes = standard_max_context * np.float64(standard_bytes_per_token)
    offloaded_gpu_bytes = offloaded_max_context * np.float64(offloaded_bytes)

    # Recompute multiplier from the two memory layouts.
    return float(
        (g / offloaded_gpu_bytes * sequence_length)
        / (g / standard_gpu_bytes)
    )


def grade(sol, fx) -> dict:
    cases = [
        (1, 1024, 2, 1_000_000_000, 2048),
        (4, 4096, 2, 8_000_000_000, 4096),
        (2, 8192, 4, 16_000_000_000, 8192),
        (8, 2048, 2, 4_000_000_000, 1024),
    ]

    errors = []
    for case in cases:
        ref = _oracle(*case)
        try:
            got = float(sol.context_multiplier(*case))
        except Exception:
            return {"rel_err": 1.0}
        errors.append(abs(got - ref) / (abs(ref) + 1e-12))

    return {"rel_err": float(np.max(np.asarray(errors)))}

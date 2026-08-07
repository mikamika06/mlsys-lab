import numpy as np


def analyze_accumulation_discrepancy(
    correct_grads: list[dict[str, np.ndarray]],
    buggy_grads: list[dict[str, np.ndarray]],
    accum_steps: int,
) -> dict[str, float]:
    if not correct_grads or not buggy_grads or len(correct_grads) != len(buggy_grads):
        return {
            "max_abs_error": 0.0,
            "effective_batch_fraction": 0.0,
            "is_buggy": 0.0,
        }

    max_err = 0.0
    for g_c, g_b in zip(correct_grads, buggy_grads):
        for k in g_c:
            err = np.max(np.abs(g_c[k] - g_b[k]))
            if err > max_err:
                max_err = float(err)

    eff_fraction = 1.0 / float(accum_steps)
    is_buggy = 1.0 if max_err > 1e-5 else 0.0

    return {
        "max_abs_error": float(max_err),
        "effective_batch_fraction": float(eff_fraction),
        "is_buggy": float(is_buggy),
    }

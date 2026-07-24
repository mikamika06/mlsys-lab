import numpy as np


def _oracle(weights, logits):
    weights = np.asarray(weights, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)

    soft_idx = np.argsort(-logits, axis=1, kind="stable")[:, :2]
    mag_idx = np.argsort(-np.abs(weights), axis=1, kind="stable")[:, :2]

    soft_mask = np.zeros_like(weights)
    mag_mask = np.zeros_like(weights)

    rows = np.arange(weights.shape[0])
    soft_mask[rows[:, None], soft_idx] = 1.0
    mag_mask[rows[:, None], mag_idx] = 1.0

    soft_retained = np.sum(np.abs(weights) * soft_mask, axis=1)
    magnitude_retained = np.sum(np.abs(weights) * mag_mask, axis=1)

    soft_error = np.sum((weights - weights * soft_mask) ** 2, axis=1)
    magnitude_error = np.sum((weights - weights * mag_mask) ** 2, axis=1)

    soft_total = np.sum(soft_error)
    mag_total = np.sum(magnitude_error)

    if soft_total < mag_total:
        better = "soft"
    elif mag_total < soft_total:
        better = "magnitude"
    else:
        better = "tie"

    return {
        "soft_retained": soft_retained,
        "magnitude_retained": magnitude_retained,
        "soft_error": soft_error,
        "magnitude_error": magnitude_error,
        "better": better,
    }


def grade(sol, fx) -> dict:
    weights = np.array(
        [
            [3.0, -1.0, 2.0, 0.5],
            [0.2, -4.0, 1.0, 3.0],
            [-2.0, 2.0, -1.0, 0.5],
            [5.0, 4.0, 3.0, 2.0],
            [1.0, 1.0, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    logits = np.array(
        [
            [0.1, 5.0, 4.0, 3.0],
            [2.0, 1.0, 7.0, 6.0],
            [9.0, 8.0, 1.0, 0.0],
            [0.0, 4.0, 3.0, 2.0],
            [1.0, 0.5, 2.0, 1.5],
        ],
        dtype=np.float64,
    )

    ref = _oracle(weights, logits)

    try:
        got = sol.compare_2_4_masks(weights, logits)
        numeric_mse = 0.0
        for key in (
            "soft_retained",
            "magnitude_retained",
            "soft_error",
            "magnitude_error",
        ):
            a = np.asarray(got[key], dtype=np.float64)
            b = np.asarray(ref[key], dtype=np.float64)
            numeric_mse += float(np.mean((a - b) ** 2))
        numeric_mse /= 4.0
        better_match = 1.0 if got["better"] == ref["better"] else 0.0
    except Exception:
        numeric_mse = float("inf")
        better_match = 0.0

    return {
        "mse": numeric_mse,
        "better_match": better_match,
    }

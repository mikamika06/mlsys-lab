import numpy as np


def analyze_near_tie_flips(logits_a, logits_b, margin):
    la = np.asarray(logits_a, dtype=np.float64)
    lb = np.asarray(logits_b, dtype=np.float64)
    if la.shape != lb.shape:
        raise ValueError("Logits sets must have identical dimensions.")

    n_positions = la.shape[0]
    argmax_a = np.argmax(la, axis=-1)
    argmax_b = np.argmax(lb, axis=-1)

    near_tie_count = 0
    flip_count = 0

    for i in range(n_positions):
        sorted_a = np.sort(la[i])
        gap_a = sorted_a[-1] - sorted_a[-2]
        if gap_a <= margin:
            near_tie_count += 1
            if argmax_a[i] != argmax_b[i]:
                flip_count += 1

    flip_fraction = (flip_count / float(near_tie_count)) if near_tie_count > 0 else 0.0

    return {
        "total_positions": int(n_positions),
        "near_tie_count": int(near_tie_count),
        "flip_count": int(flip_count),
        "flip_fraction": float(flip_fraction)
    }

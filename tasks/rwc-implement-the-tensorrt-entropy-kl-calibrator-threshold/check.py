import numpy as np


def _oracle(hist):
    hist = np.asarray(hist, dtype=np.float64)

    def kl_value(t):
        clipped = hist[:t + 1].copy()
        if t + 1 < hist.size:
            clipped[-1] += np.sum(hist[t + 1:])

        p = clipped / np.sum(clipped)

        q_counts = np.zeros_like(clipped)
        edges = np.linspace(0, t + 1, 129, dtype=np.int64)
        for i in range(128):
            start = int(edges[i])
            end = int(edges[i + 1])
            if end <= start:
                continue
            q_counts[start:end] = np.sum(clipped[start:end]) / (end - start)

        q = q_counts / np.sum(q_counts)
        mask = p > 0
        return float(np.sum(p[mask] * np.log((p[mask] + 1e-12) / (q[mask] + 1e-12))))

    curve = np.asarray(
        [kl_value(t) for t in range(128, 2048)],
        dtype=np.float64,
    )
    return int(np.argmin(curve) + 128), curve


def grade(sol, fx) -> dict:
    cases = [
        np.concatenate(
            [np.arange(1, 65, dtype=np.float64), np.zeros(1984)]
        ),
        (np.exp(-np.arange(2048) / 120.0) * 1000.0).astype(np.float64),
        (np.sin(np.arange(2048) / 20.0) ** 2 * 500.0 + 1.0).astype(np.float64),
    ]

    argmin_ok = 1.0
    max_err = 0.0

    for hist in cases:
        expected_t, expected_curve = _oracle(hist)
        try:
            got_t, got_curve = sol.entropy_calibration_threshold(hist)
            got_curve = np.asarray(got_curve, dtype=np.float64)
        except Exception:
            argmin_ok = 0.0
            max_err = float("inf")
            continue

        if int(got_t) != expected_t:
            argmin_ok = 0.0

        err = np.linalg.norm(got_curve - expected_curve) / (
            np.linalg.norm(expected_curve) + 1e-12
        )
        max_err = max(max_err, float(err))

    return {
        "argmin_index": argmin_ok,
        "kl_curve_rel_err": float(max_err),
    }

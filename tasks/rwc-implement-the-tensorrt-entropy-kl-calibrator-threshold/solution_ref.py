import numpy as np


def entropy_calibration_threshold(hist: np.ndarray) -> tuple[int, np.ndarray]:
    hist = np.asarray(hist, dtype=np.float64)
    values = []

    for t in range(128, 2048):
        clipped = hist[:t + 1].copy()
        if t + 1 < hist.size:
            clipped[-1] += np.sum(hist[t + 1:])

        p = clipped / np.sum(clipped)

        q_counts = np.zeros_like(clipped)
        edges = np.linspace(0, t + 1, 129, dtype=np.int64)

        for i in range(128):
            start = int(edges[i])
            end = int(edges[i + 1])
            if end > start:
                q_counts[start:end] = np.sum(clipped[start:end]) / (end - start)

        q = q_counts / np.sum(q_counts)

        mask = p > 0
        values.append(
            np.sum(
                p[mask]
                * np.log((p[mask] + 1e-12) / (q[mask] + 1e-12))
            )
        )

    curve = np.asarray(values, dtype=np.float64)
    return int(np.argmin(curve) + 128), curve

import numpy as np


def _oracle(W, ratios, bits):
    W = np.asarray(W, dtype=np.float64)
    ratios = np.asarray(ratios, dtype=np.float64)

    qmax = (1 << (bits - 1)) - 1
    max_abs = np.max(np.abs(W), axis=1)

    curve = []
    for ratio in ratios:
        bounds = (max_abs * ratio)[:, None]
        scales = bounds / qmax
        clipped = np.clip(W, -bounds, bounds)
        quantized = np.round(clipped / scales)
        reconstructed = quantized * scales
        curve.append(np.mean((W - reconstructed) ** 2))

    curve = np.asarray(curve, dtype=np.float64)
    return int(np.argmin(curve)), curve


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, -2.0, 0.5], [3.0, -1.0, 2.0]], dtype=np.float64),
            np.array([0.5, 0.625, 0.75, 0.875, 1.0]),
            4,
        ),
        (
            np.array(
                [
                    [0.2, -1.7, 2.8, 1.1],
                    [-3.2, 0.5, 1.4, -2.1],
                    [4.5, -0.2, 0.3, 1.9],
                ],
                dtype=np.float64,
            ),
            np.linspace(0.5, 1.0, 9),
            3,
        ),
        (
            np.array([[0.1, -0.4, 0.8], [2.0, -2.5, 1.2]], dtype=np.float64),
            np.array([0.5, 0.7, 0.9, 1.0]),
            8,
        ),
    ]

    index_ok = 1.0
    curve_ok = 1.0

    for W, ratios, bits in cases:
        ref_idx, ref_curve = _oracle(W, ratios, bits)
        try:
            got_idx, got_curve = sol.search_clip_ratio(W, ratios, bits)
        except Exception:
            return {"argmin_index": 0.0, "mse_curve": 0.0}

        if int(got_idx) != ref_idx:
            index_ok = 0.0

        if not np.allclose(
            np.asarray(got_curve, dtype=np.float64),
            ref_curve,
            rtol=1e-12,
            atol=1e-12,
        ):
            curve_ok = 0.0

    return {
        "argmin_index": index_ok,
        "mse_curve": curve_ok,
    }

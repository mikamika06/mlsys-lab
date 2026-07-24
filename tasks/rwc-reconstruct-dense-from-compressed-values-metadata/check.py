import numpy as np


def _compress_oracle(weight):
    flat = np.asarray(weight, dtype=np.float64).ravel()
    values = []
    positions = []
    metadata = []
    compressed = np.zeros_like(flat)

    for start in range(0, flat.size, 4):
        block = flat[start:start + 4]
        idx = np.argsort(-np.abs(block))[:2]
        idx = np.sort(idx)

        values.extend(block[idx])
        positions.extend(idx.tolist())

        compressed[start + idx[0]] = block[idx[0]]
        compressed[start + idx[1]] = block[idx[1]]

        metadata.append(int(idx[0]) | (int(idx[1]) << 2))

    return (
        np.asarray(values, dtype=np.float64),
        np.asarray(metadata, dtype=np.uint8),
        np.asarray(positions, dtype=np.int64),
        compressed.reshape(weight.shape),
    )


def grade(sol, fx) -> dict:
    cases = [
        np.array([[1.5, -0.2, 0.0, -2.0], [4.0, 3.0, 0.5, 0.1]], dtype=np.float64),
        np.arange(-16, 16, dtype=np.float64).reshape(4, 8) / 3.0,
        np.array(
            [[0.1, -5.0, 2.0, 0.0, 7.0, 1.0, -3.0, 4.0]],
            dtype=np.float64,
        ),
    ]

    max_err = 0.0
    positions_ok = 1.0

    for weight in cases:
        values, metadata, ref_positions, ref_dense = _compress_oracle(weight)

        try:
            dense, positions = sol.reconstruct_dense(values, metadata, weight.shape)
            dense = np.asarray(dense, dtype=np.float64)
            positions = np.asarray(positions, dtype=np.int64)
        except Exception:
            return {"max_abs_err": float("inf"), "positions_exact": 0.0}

        if dense.shape != ref_dense.shape:
            max_err = float("inf")
        else:
            max_err = max(max_err, float(np.max(np.abs(dense - ref_dense))))

        if not np.array_equal(positions, ref_positions):
            positions_ok = 0.0

    return {
        "max_abs_err": max_err,
        "positions_exact": positions_ok,
    }

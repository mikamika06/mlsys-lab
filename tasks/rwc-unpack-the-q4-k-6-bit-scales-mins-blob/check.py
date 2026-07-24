import numpy as np


def _get_scale_min_k4(j: int, q: np.ndarray):
    """Faithful port of ggml's get_scale_min_k4 (used by dequantize_row_q4_K)."""
    q = np.asarray(q, dtype=np.uint8)
    if j < 4:
        d = int(q[j] & 63)
        m = int(q[j + 4] & 63)
    else:
        d = int((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4))
        m = int((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4))
    return d, m


def _oracle_unpack(blob: np.ndarray):
    blob = np.asarray(blob, dtype=np.uint8)
    scales = np.zeros(8, dtype=np.uint8)
    mins = np.zeros(8, dtype=np.uint8)
    for j in range(8):
        d, m = _get_scale_min_k4(j, blob)
        scales[j] = d
        mins[j] = m
    return scales, mins


def _pack(scales, mins) -> np.ndarray:
    """Forward packer (inverse of get_scale_min_k4) used only to build test
    blobs from known ground-truth 6-bit scale/min values."""
    q = [0] * 12
    for j in range(4):
        q[j] |= scales[j] & 63
        q[j + 4] |= mins[j] & 63
    for j in range(4, 8):
        q[j + 4] = (scales[j] & 0x0F) | ((mins[j] & 0x0F) << 4)
        q[j - 4] |= ((scales[j] >> 4) & 0x3) << 6
        q[j] |= ((mins[j] >> 4) & 0x3) << 6
    return np.array(q, dtype=np.uint8)


def _scenarios(rng: np.random.Generator):
    blobs = []

    # All-zero and all-max (63) round-trip blobs.
    blobs.append(_pack([0] * 8, [0] * 8))
    blobs.append(_pack([63] * 8, [63] * 8))

    # Random 6-bit ground-truth scale/min vectors, packed then unpacked.
    for _ in range(4):
        scales = rng.integers(0, 64, size=8).tolist()
        mins = rng.integers(0, 64, size=8).tolist()
        blobs.append(_pack(scales, mins))

    # Raw random 12-byte blobs (not necessarily produced by _pack, but
    # get_scale_min_k4 must still decode them consistently).
    for _ in range(3):
        blobs.append(rng.integers(0, 256, size=12, dtype=np.uint8))

    return blobs


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0
    for blob in _scenarios(rng):
        scales_ref, mins_ref = _oracle_unpack(blob)
        try:
            scales_got, mins_got = sol.unpack_q4k_scales_mins(blob.copy())
            scales_got = np.asarray(scales_got)
            mins_got = np.asarray(mins_got)
        except Exception:
            return {"exact_match": 0.0}

        if scales_got.shape != (8,) or mins_got.shape != (8,):
            return {"exact_match": 0.0}
        if not np.array_equal(scales_got, scales_ref) or not np.array_equal(mins_got, mins_ref):
            return {"exact_match": 0.0}

    return {"exact_match": ok}

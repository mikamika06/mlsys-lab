import numpy as np


def _oracle_decode_one(code):
    sign = -1.0 if (code & 0x80) else 1.0
    e = (code >> 3) & 0x0F
    m = code & 0x07
    if e == 0:
        return sign * (m / 8.0) * (2.0 ** -6)
    if e == 15 and m == 7:
        return np.nan
    return sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))


def _oracle_table():
    vals = np.empty(256, dtype=np.float64)
    for i in range(256):
        vals[i] = _oracle_decode_one(i)
    return vals


def _oracle_encode(arr):
    table = _oracle_table()
    out = np.empty(arr.shape, dtype=np.uint8)
    for idx, value in np.ndenumerate(arr.astype(np.float64)):
        if np.isnan(value):
            value = 0.0
        sign = 0x80 if value < 0 else 0
        mag = abs(value)
        candidates = []
        for code in range(256):
            if code & 0x80 != sign:
                continue
            if (code & 0x7F) == 0x7F:
                continue
            candidates.append(code)
        best = candidates[0]
        best_dist = abs(abs(table[best]) - mag)
        for code in candidates[1:]:
            dist = abs(abs(table[code]) - mag)
            if dist < best_dist:
                best = code
                best_dist = dist
            elif dist == best_dist:
                if (code & 0x07) % 2 == 0 and (best & 0x07) % 2 == 1:
                    best = code
        out[idx] = best
    return out


def grade(sol, fx) -> dict:
    cases = np.array(
        [
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5,
            2.0 ** -10, 2.0 ** -9, 2.0 ** -6,
            0.0001, -0.0001,
            440.0, 448.0, 449.0, 500.0, -1000.0,
            1.0625, 1.1875, 3.75, -3.75,
        ],
        dtype=np.float32,
    )
    try:
        got_codes = np.asarray(sol.encode_e4m3fn(cases), dtype=np.uint8)
        ref_codes = _oracle_encode(cases)
        exact = float(np.array_equal(got_codes, ref_codes))

        decoded = np.asarray(sol.decode_e4m3fn(ref_codes), dtype=np.float64)
        ref_decoded = _oracle_decode_one
        expected = np.array([ref_decoded(int(x)) for x in ref_codes], dtype=np.float64)
        finite = np.isfinite(expected)
        err = float(np.max(np.abs(decoded[finite] - expected[finite])))
    except Exception:
        return {"exact_match": 0.0, "max_abs_err": float("inf")}

    return {"exact_match": exact, "max_abs_err": err}

import numpy as np


def _ref_ulp_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    def ordered_ints(x):
        bits = x.view(np.uint32)
        sign = (bits & np.uint32(0x80000000)) != 0
        return np.where(
            sign,
            np.bitwise_not(bits),
            bits ^ np.uint32(0x80000000),
        ).astype(np.int64)

    ka = ordered_ints(a)
    kb = ordered_ints(b)
    return np.abs(ka - kb).astype(np.uint32)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array(
                [
                    -0.0,
                    0.0,
                    -np.float32("1e-45"),
                    np.float32("1e-45"),
                    -1.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    0.0,
                    np.nextafter(np.float32(0.0), np.float32(1.0)),
                    0.0,
                    np.float32("2e-45"),
                    -0.99999994,
                    np.nextafter(np.float32(1.0), np.float32(2.0)),
                ],
                dtype=np.float32,
            ),
        ),
        (
            np.array(
                [
                    np.float32(1.0),
                    np.float32(2.0),
                    np.float32(65536.0),
                    np.float32(-65536.0),
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    np.nextafter(np.float32(1.0), np.float32(2.0)),
                    np.nextafter(np.float32(2.0), np.float32(3.0)),
                    np.nextafter(np.float32(65536.0), np.float32(65537.0)),
                    np.nextafter(np.float32(-65536.0), np.float32(-65537.0)),
                ],
                dtype=np.float32,
            ),
        ),
    ]

    ok = 1.0
    for a, b in cases:
        try:
            got = sol.ulp_distance(a, b)
        except Exception:
            ok = 0.0
            break
        ref = _ref_ulp_distance(a, b)
        if not isinstance(got, np.ndarray) or got.dtype != np.uint32 or not np.array_equal(got, ref):
            ok = 0.0
            break

    return {"exact_match": ok}

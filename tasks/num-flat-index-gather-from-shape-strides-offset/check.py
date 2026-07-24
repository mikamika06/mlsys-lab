import numpy as np

from mlsys.scorers import byte_exact_fraction


def _cases(rng):
    """(buf, shape, strides, offset) cases covering reshape / transpose /
    offset sub-block / zero-stride broadcast / negative-stride reversal.
    All strides are in BYTES, offset is in ELEMENTS, matching flat_gather's contract.
    """
    s = np.dtype(np.float64).itemsize  # 8
    cases = []

    # plain reshape (4, 6)
    buf = rng.standard_normal(24)
    cases.append((buf, (4, 6), (6 * s, s), 0))

    # transpose of the same logical (4, 6) block
    buf2 = rng.standard_normal(24)
    cases.append((buf2, (6, 4), (s, 6 * s), 0))

    # offset sub-block: a (3, 4) window into a logical (6, 5) row-major array,
    # starting at row 1, col 1
    buf3 = rng.standard_normal(30)
    cases.append((buf3, (3, 4), (5 * s, s), 6))

    # zero-stride broadcast: repeat a 5-element run across 4 rows
    buf4 = rng.standard_normal(30)
    cases.append((buf4, (4, 5), (0, s), 3))

    # negative-stride reversal of a 6-element run
    buf5 = rng.standard_normal(20)
    cases.append((buf5, (6,), (-s,), 5))

    # 3-D case with a broadcast middle axis
    buf6 = rng.standard_normal(60)
    cases.append((buf6, (3, 4, 5), (20 * s, 0, s), 0))

    return cases


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    fracs = []
    for buf, shape, strides, offset in _cases(rng):
        buf = np.asarray(buf, dtype=np.float64)
        ref = np.lib.stride_tricks.as_strided(
            buf[offset:], shape=shape, strides=strides
        ).copy()

        try:
            got = sol.flat_gather(buf, shape, strides, offset)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            fracs.append(0.0)
            continue

        if got.shape != ref.shape:
            fracs.append(0.0)
            continue

        fracs.append(byte_exact_fraction(ref, got))

    return {"byte_exact_fraction": float(min(fracs))}

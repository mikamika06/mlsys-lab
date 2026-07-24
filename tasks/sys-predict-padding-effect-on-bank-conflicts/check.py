import numpy as np

_WARP = 32
_SEARCH_RANGE = 8   # pad in [0, _SEARCH_RANGE) always suffices to hit degree 1


def _degree(stride):
    """Max lanes-per-bank for a warp doing address t*stride, t=0..31."""
    banks = [0] * _WARP
    for t in range(_WARP):
        banks[(t * stride) % _WARP] += 1
    return max(banks)


def _min_degree(width):
    return min(_degree(width + p) for p in range(_SEARCH_RANGE))


def grade(sol, fx) -> dict:
    """
    For a set of tile widths (fixed edge cases + seeded random values),
    computes the true minimum achievable bank-conflict degree by brute
    force over a small padding search range, and checks that the
    submission's choose_padding(width) achieves that same minimum degree
    under the 32-bank column-stride model, for every width.
    """
    rng = np.random.default_rng(0)
    widths = [1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 101, 127, 128]
    widths += [int(x) for x in rng.integers(1, 500, size=8)]

    ok = 1.0
    for width in widths:
        min_deg = _min_degree(width)
        try:
            pad = int(sol.choose_padding(width))
        except Exception:
            ok = 0.0
            break
        if pad < 0:
            ok = 0.0
            break
        if _degree(width + pad) != min_deg:
            ok = 0.0
            break
    return {"modeled_mem_access": ok}

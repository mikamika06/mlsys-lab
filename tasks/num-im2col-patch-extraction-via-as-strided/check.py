import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from mlsys import scorers


def _oracle(x: np.ndarray, kh: int, kw: int, stride: int) -> np.ndarray:
    full = sliding_window_view(x, (kh, kw))  # (H-kh+1, W-kw+1, kh, kw), real view
    return full[::stride, ::stride]


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    byte_fracs = []
    view_oks = []
    for _ in range(6):
        H = int(rng.integers(6, 16))
        W = int(rng.integers(6, 16))
        kh = int(rng.integers(2, min(5, H) + 1))
        kw = int(rng.integers(2, min(5, W) + 1))
        stride = int(rng.integers(1, 4))
        x = rng.standard_normal((H, W)).astype(np.float64)

        ref = _oracle(x, kh, kw, stride)

        try:
            got = sol.im2col_patches(x, kh, kw, stride)
        except Exception:
            byte_fracs.append(0.0)
            view_oks.append(0.0)
            continue

        got_arr = np.asarray(got)
        if got_arr.shape != ref.shape:
            byte_fracs.append(0.0)
            view_oks.append(0.0)
            continue

        byte_fracs.append(scorers.byte_exact_fraction(got_arr, ref))
        try:
            view_oks.append(1.0 if np.shares_memory(got, x) else 0.0)
        except Exception:
            view_oks.append(0.0)

    return {
        "byte_exact_fraction": min(byte_fracs) if byte_fracs else 0.0,
        "is_view": min(view_oks) if view_oks else 0.0,
    }

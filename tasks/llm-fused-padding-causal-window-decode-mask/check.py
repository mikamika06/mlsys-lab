import numpy as np

def _reference(padding_mask: np.ndarray, window_size: int) -> np.ndarray:
    B, T = padding_mask.shape
    rows = np.arange(T).reshape(-1, 1)
    cols = np.arange(T).reshape(1, -1)
    # Windowed causal mask: j <= i and i-j < window_size
    window = (cols <= rows) & (rows - cols < window_size)
    # Broadcast across batch
    mask = window[None, :, :] & padding_mask[:, None, :]
    # Zero out rows where target is padded
    mask[~padding_mask[:, :, None]] = False
    return mask

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0
    for _ in range(10):
        B = rng.integers(1, 5)
        T = rng.integers(5, 21)
        w = rng.integers(1, T + 1)
        pad = rng.choice([False, True], size=(B, T), p=[0.2, 0.8])
        try:
            got = sol.fused_decode_mask(pad, int(w))
        except Exception:
            ok = 0.0
            break
        if not isinstance(got, np.ndarray):
            ok = 0.0
            break
        ref = _reference(pad, int(w))
        if got.shape != ref.shape:
            ok = 0.0
            break
        err = float(np.max(np.abs(got.astype(int) - ref.astype(int))))
        if err > 0:
            ok = 0.0
            break
    return {"max_abs_err": ok}

import numpy as np

def grade(sol, fx):
    """Grade by comparing student scales against oracle-computed scales."""
    rng = np.random.RandomState(42)
    
    cases = [
        (rng.randn(4, 3).astype(np.float64), rng.randn(3, 5).astype(np.float64)),
        (rng.randn(8, 16).astype(np.float64), rng.randn(16, 8).astype(np.float64)),
        (rng.randn(1, 4).astype(np.float64), rng.randn(4, 1).astype(np.float64)),
        (np.abs(rng.randn(10, 10).astype(np.float64)), np.abs(rng.randn(10, 10).astype(np.float64))),
    ]
    
    worst_rx = 0.0
    worst_rw = 0.0
    
    for X, W in cases:
        scale_x, scale_w = sol.compute_int8_scales(X, W)
        scale_x = np.asarray(scale_x, dtype=np.float64).ravel()
        scale_w = np.asarray(scale_w, dtype=np.float64).ravel()

        # Oracle reference
        ref_x = np.max(np.abs(X), axis=1).astype(np.float64) / 127.0
        ref_w = np.max(np.abs(W), axis=0).astype(np.float64) / 127.0

        # Relative L2 error
        norm_x = np.linalg.norm(ref_x)
        norm_w = np.linalg.norm(ref_w)

        rx = float(np.linalg.norm(scale_x - ref_x) / norm_x) if norm_x > 1e-12 else 0.0
        rw = float(np.linalg.norm(scale_w - ref_w) / norm_w) if norm_w > 1e-12 else 0.0

        worst_rx = max(worst_rx, rx)
        worst_rw = max(worst_rw, rw)

    return {"rel_err_x": worst_rx, "rel_err_w": worst_rw}

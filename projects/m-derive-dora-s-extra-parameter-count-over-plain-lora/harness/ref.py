import numpy as np


def get_test_cases():
    np.random.seed(42)
    cases = []
    for _ in range(5):
        d_in = int(np.random.randint(32, 128))
        d_out = int(np.random.randint(32, 128))
        r = int(np.random.randint(4, 16))
        alpha = float(np.random.choice([8.0, 16.0, 32.0]))
        w = np.random.randn(d_out, d_in).astype(np.float32)
        a = np.random.randn(r, d_in).astype(np.float32)
        b = np.random.randn(d_out, r).astype(np.float32)
        g = np.random.randn(d_out).astype(np.float32)
        x = np.random.randn(16, d_in).astype(np.float32)
        out = {
            "d_in": d_in,
            "d_out": d_out,
            "r": r,
            "alpha": alpha,
            "extra_params": d_out,
            "w": w,
            "a": a,
            "b": b,
            "g": g,
            "x": x,
        }
        cases.append(out)
    return cases

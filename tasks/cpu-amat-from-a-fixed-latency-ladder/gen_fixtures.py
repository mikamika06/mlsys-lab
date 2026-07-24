"""Generate fixture file for AMAT computation tests."""
def _gen_hit_rates() -> "np.ndarray":
    import numpy as np
    return np.array([
        [0.98, 0.95, 0.90],
        [1.00, 1.00, 1.00],
        [0.70, 0.80, 0.85],
        [0.60, 0.65, 0.70],
        [0.90, 0.92, 0.93]
    ], dtype=np.float64)

if __name__ == "__main__":
    import numpy as np
    hit_rates = _gen_hit_rates()
    np.save('hit_rates.npy', hit_rates)

import numpy as np


def grade(sol, fx) -> dict:
    """Builds inputs that are adversarial for a single-accumulator
    sequential float32 reduction: one large-magnitude term followed by many
    terms individually smaller than half the accumulator's float32 ULP at
    that magnitude. Under strict IEEE round-to-nearest, any such term added
    on its own is a complete no-op (x + tiny == x exactly), so a naive
    sequential reduction silently drops nearly all of them, while a
    pairwise/tree reduction sums the small terms among themselves first
    (where they are not swamped) and only merges with the large term once.

    The oracle is the exact float64 sum of the given float32 values
    (`np.sum` on the float64 cast), never the candidate's own algorithm.
    One plain, non-adversarial random case is included too, so a solution
    can't special-case the pathological structure.
    """
    rng = np.random.default_rng(0)
    rel_errs = []

    # Adversarial trials: 1 huge term + many terms below half its ULP.
    for _ in range(4):
        n_small = int(rng.integers(5000, 20000))
        huge = float(rng.uniform(5e7, 5e8))
        huge32 = np.float32(huge)
        ulp = float(np.spacing(huge32))
        small_max = 0.4 * ulp
        smalls = rng.uniform(0.05 * ulp, small_max, size=n_small).astype(np.float32)
        x = np.concatenate([[huge32], smalls]).astype(np.float32)
        w = np.ones_like(x, dtype=np.float32)

        oracle = float(np.sum(x.astype(np.float64) * w.astype(np.float64)))
        try:
            got = float(sol.fused_dot_reduce(x, w))
        except Exception:
            return {"rel_err": float("inf")}
        rel_errs.append(abs(got - oracle) / (abs(oracle) + 1e-12))

    # Plain, non-adversarial correctness check.
    for _ in range(2):
        n = int(rng.integers(200, 800))
        x = rng.standard_normal(n).astype(np.float32) * 10.0
        w = rng.standard_normal(n).astype(np.float32)
        oracle = float(np.sum(x.astype(np.float64) * w.astype(np.float64)))
        try:
            got = float(sol.fused_dot_reduce(x, w))
        except Exception:
            return {"rel_err": float("inf")}
        rel_errs.append(abs(got - oracle) / (abs(oracle) + 1e-12))

    return {"rel_err": float(max(rel_errs))}

import numpy as np

from mlsys import scorers

TILE_POOL = [
    (32, 32), (32, 64), (64, 32), (64, 64), (64, 128), (128, 64),
    (128, 128), (128, 256), (256, 128), (256, 64), (64, 256),
    (32, 128), (128, 32), (256, 256),
]
SM_COUNTS = [56, 68, 80, 82, 108, 132]

N_CASES = 40


def _cost_vector(M: int, N: int, num_SMs: int, candidates) -> np.ndarray:
    costs = np.empty(len(candidates), dtype=np.float64)
    for i, (BM, BN) in enumerate(candidates):
        ctas_m = -(-M // BM)
        ctas_n = -(-N // BN)
        CTAs = ctas_m * ctas_n
        waves = -(-CTAs // num_SMs)
        tile_area = (ctas_m * BM) * (ctas_n * BN)
        waste = tile_area - M * N
        costs[i] = waves + waste / tile_area
    return costs


def _oracle(M: int, N: int, num_SMs: int, candidates):
    costs = _cost_vector(M, N, num_SMs, candidates)
    return int(np.argmin(costs)), costs


def _gen_case(rng: np.random.Generator):
    M = int(rng.integers(17, 4096))
    N = int(rng.integers(17, 4096))
    K = int(rng.integers(32, 4096))
    num_SMs = int(rng.choice(SM_COUNTS))
    n_cand = int(rng.integers(4, 8))
    idxs = rng.choice(len(TILE_POOL), size=n_cand, replace=False)
    candidates = [TILE_POOL[i] for i in idxs]
    return M, N, K, num_SMs, candidates


def _fail() -> dict:
    return {"argmin_agreement": 0.0, "cost_rel_err": float("inf")}


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    agree = 0
    ref_all, got_all = [], []

    for _ in range(N_CASES):
        M, N, K, num_SMs, candidates = _gen_case(rng)
        ref_idx, ref_costs = _oracle(M, N, num_SMs, candidates)

        try:
            out = sol.select_autotune_tile(M, N, K, num_SMs, candidates)
            got_idx, got_costs = out[0], out[1]
            got_idx = int(got_idx)
            got_costs = np.asarray(got_costs, dtype=np.float64).ravel()
        except Exception:
            return _fail()

        if got_costs.shape != ref_costs.shape:
            return _fail()

        if got_idx == ref_idx:
            agree += 1
        ref_all.append(ref_costs)
        got_all.append(got_costs)

    ref_cat = np.concatenate(ref_all)
    got_cat = np.concatenate(got_all)

    return {
        "argmin_agreement": agree / N_CASES,
        "cost_rel_err": scorers.rel_err(ref_cat, got_cat),
    }

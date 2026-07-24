import numpy as np


def _group_quant_mse(W: np.ndarray, group_size: int, bits: int) -> float:
    qmax = (1 << (bits - 1)) - 1
    n = W.shape[0]
    blocks = W.reshape(n // group_size, group_size)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    scale = np.where(amax > 0, amax / qmax, 1.0)
    codes = np.clip(np.round(blocks / scale), -qmax, qmax)
    recon = codes * scale
    return float(np.mean((recon - blocks) ** 2))


def _oracle(W, group_sizes, bits, lam):
    W = np.asarray(W, dtype=np.float64)
    costs = np.zeros(len(group_sizes), dtype=np.float64)
    for i, gs in enumerate(group_sizes):
        mse = _group_quant_mse(W, gs, bits)
        overhead = 16.0 / gs
        costs[i] = mse + lam * overhead
    best_idx = int(np.argmin(costs))
    return int(group_sizes[best_idx]), float(costs[best_idx]), costs


def _build_cases():
    cases = []
    group_sizes = (32, 64, 128, 256)
    for seed, n, bits, lam in [(0, 1024, 4, 0.02), (1, 2048, 4, 0.005), (2, 1024, 3, 0.05)]:
        rng = np.random.default_rng(seed)
        W = rng.standard_normal(n)
        cases.append((W, group_sizes, bits, lam))
    return cases


def grade(sol, fx) -> dict:
    argmin_ok = 1.0
    worst_cost_err = 0.0

    for W, group_sizes, bits, lam in _build_cases():
        best_gs_ref, _best_cost_ref, costs_ref = _oracle(W, group_sizes, bits, lam)

        try:
            got = sol.pick_int4_group_size(W.copy(), group_sizes, bits=bits, lam=lam)
            best_gs_got, _best_cost_got, costs_got = got
            costs_got = np.asarray(costs_got, dtype=np.float64)
        except Exception:
            return {"argmin_match": 0.0, "cost_max_abs_err": float("inf")}

        if costs_got.shape != costs_ref.shape or not np.all(np.isfinite(costs_got)):
            return {"argmin_match": 0.0, "cost_max_abs_err": float("inf")}

        if int(best_gs_got) != best_gs_ref:
            argmin_ok = 0.0

        worst_cost_err = max(worst_cost_err, float(np.max(np.abs(costs_got - costs_ref))))

    return {"argmin_match": argmin_ok, "cost_max_abs_err": worst_cost_err}

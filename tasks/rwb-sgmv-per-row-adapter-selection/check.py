import numpy as np


def _oracle(x, adapter_id, A_bank, B_bank, scale):
    N = x.shape[0]
    d_out = B_bank.shape[2]
    out = np.empty((N, d_out), dtype=np.float64)
    for i in range(N):
        aid = int(adapter_id[i])
        delta = (x[i] @ A_bank[aid]) @ B_bank[aid]
        out[i] = scale[aid] * delta
    return out


def _synthetic_cases():
    rng = np.random.default_rng(97)
    cases = []
    for _ in range(4):
        N = int(rng.integers(5, 20))
        d_in = int(rng.integers(3, 10))
        d_out = int(rng.integers(3, 10))
        r = int(rng.integers(1, 5))
        num_adapters = int(rng.integers(2, 6))

        x = rng.standard_normal((N, d_in))
        adapter_id = rng.integers(0, num_adapters, size=N).astype(np.int64)
        A_bank = rng.standard_normal((num_adapters, d_in, r)) * 0.5
        B_bank = rng.standard_normal((num_adapters, r, d_out)) * 0.5
        scale = rng.uniform(0.5, 2.0, size=num_adapters)
        cases.append((x, adapter_id, A_bank, B_bank, scale))
    return cases


def grade(sol, fx) -> dict:
    cases = [(fx["x"], fx["adapter_id"], fx["A_bank"], fx["B_bank"], fx["scale"])] + _synthetic_cases()

    worst = 0.0
    for x, adapter_id, A_bank, B_bank, scale in cases:
        ref = _oracle(x, adapter_id, A_bank, B_bank, scale)
        try:
            got = np.asarray(
                sol.sgmv_apply(x.copy(), adapter_id.copy(), A_bank.copy(), B_bank.copy(), scale.copy()),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}

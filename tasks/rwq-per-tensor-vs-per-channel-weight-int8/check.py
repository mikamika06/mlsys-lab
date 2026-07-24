import numpy as np


def _sym_int8_quant(g: np.ndarray) -> np.ndarray:
    amax = float(np.max(np.abs(g)))
    scale = amax / 127.0 if amax > 0 else 1.0
    codes = np.clip(np.round(g / scale), -127, 127)
    return codes * scale


def _oracle(W: np.ndarray):
    W = np.asarray(W, dtype=np.float64)

    W_hat_pt = _sym_int8_quant(W.reshape(-1)).reshape(W.shape)
    mse_pt = float(np.mean((W_hat_pt - W) ** 2))

    W_hat_pc = np.empty_like(W)
    for i in range(W.shape[0]):
        W_hat_pc[i] = _sym_int8_quant(W[i])
    mse_pc = float(np.mean((W_hat_pc - W) ** 2))

    return mse_pt, mse_pc


def _build_cases():
    cases = []
    for seed, rows, cols, mags in [
        (0, 6, 32, [100.0, 0.5, 10.0, 3.0, 0.1, 50.0]),
        (1, 4, 16, [1.0, 200.0, 0.05, 5.0]),
        (2, 8, 8, [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]),
    ]:
        rng = np.random.default_rng(seed)
        rows_data = [rng.standard_normal(cols) * m for m in mags]
        W = np.vstack(rows_data)
        cases.append(W)
    return cases


def grade(sol, fx) -> dict:
    worst_pt = 0.0
    worst_pc = 0.0
    worst_margin = float("inf")

    for W in _build_cases():
        mse_pt_ref, mse_pc_ref = _oracle(W)

        try:
            got = sol.int8_mse_per_tensor_vs_per_channel(W.copy())
            mse_pt_got, mse_pc_got = float(got[0]), float(got[1])
        except Exception:
            return {"per_tensor_abs_err": float("inf"), "per_channel_abs_err": float("inf"), "order_margin": float("-inf")}

        if not (np.isfinite(mse_pt_got) and np.isfinite(mse_pc_got)):
            return {"per_tensor_abs_err": float("inf"), "per_channel_abs_err": float("inf"), "order_margin": float("-inf")}

        worst_pt = max(worst_pt, abs(mse_pt_got - mse_pt_ref))
        worst_pc = max(worst_pc, abs(mse_pc_got - mse_pc_ref))
        worst_margin = min(worst_margin, mse_pt_got - mse_pc_got)

    return {"per_tensor_abs_err": worst_pt, "per_channel_abs_err": worst_pc, "order_margin": worst_margin}

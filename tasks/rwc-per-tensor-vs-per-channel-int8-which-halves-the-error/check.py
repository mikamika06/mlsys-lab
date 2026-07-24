import numpy as np

from mlsys import scorers


def _sym_int8_quant(x: np.ndarray, axis) -> np.ndarray:
    """Symmetric INT8 quantize/dequantize. axis=None -> one scale for the
    whole tensor (per-tensor). axis=1 -> one scale per row (per-channel)."""
    x = np.asarray(x, dtype=np.float64)
    if axis is None:
        amax = np.max(np.abs(x))
        scale = amax / 127.0 if amax > 0 else 1.0
    else:
        amax = np.max(np.abs(x), axis=axis, keepdims=True)
        scale = np.where(amax > 0, amax / 127.0, 1.0)
    q = np.clip(np.round(x / scale), -127, 127)
    return q * scale


def _oracle(W: np.ndarray):
    W = np.asarray(W, dtype=np.float64)
    W_tensor = _sym_int8_quant(W, axis=None)
    W_channel = _sym_int8_quant(W, axis=1)
    mse_tensor = scorers.mse(W, W_tensor)
    mse_channel = scorers.mse(W, W_channel)
    winner = "per_tensor" if mse_tensor < mse_channel else "per_channel"
    return mse_tensor, mse_channel, winner


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []

    # Rows with wildly different magnitude scales -> per-channel clearly wins.
    for rows, cols, spread in [(5, 8, 200.0), (3, 6, 1000.0), (8, 4, 50.0)]:
        base = rng.normal(size=(rows, cols))
        per_row_scale = rng.uniform(0.05, spread, size=(rows, 1))
        scenarios.append(base * per_row_scale)

    # All rows share the same magnitude range -> both schemes near-tied,
    # but the real per-row amax still (weakly) determines the winner.
    base = rng.normal(size=(6, 10)) * 5.0
    scenarios.append(base)

    # A single dominant outlier row alongside quiet rows.
    W = rng.normal(size=(6, 12)) * 0.1
    W[2] *= 500.0
    scenarios.append(W)

    return scenarios


def grade(sol, fx) -> dict:
    worst_rel_err = 0.0
    correct = []

    for W in _scenarios():
        mse_tensor_ref, mse_channel_ref, winner_ref = _oracle(W)

        try:
            out = sol.quant_granularity_errors(W.copy())
            mse_tensor_got = float(out["mse_per_tensor"])
            mse_channel_got = float(out["mse_per_channel"])
            winner_got = out["winner"]
        except Exception:
            return {"rel_err": float("inf"), "winner_correct": 0.0}

        ref_vec = np.array([mse_tensor_ref, mse_channel_ref], dtype=np.float64)
        got_vec = np.array([mse_tensor_got, mse_channel_got], dtype=np.float64)
        if not np.all(np.isfinite(got_vec)):
            return {"rel_err": float("inf"), "winner_correct": 0.0}

        err = scorers.rel_err(ref_vec, got_vec)
        worst_rel_err = max(worst_rel_err, err)
        correct.append(1.0 if winner_got == winner_ref else 0.0)

    return {
        "rel_err": worst_rel_err,
        "winner_correct": float(min(correct)) if correct else 0.0,
    }

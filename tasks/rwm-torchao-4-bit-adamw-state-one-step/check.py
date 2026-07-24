import numpy as np

from mlsys import scorers


def _quant4(x: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    nb = n // block_size
    xb = x.reshape(nb, block_size)
    scales = np.max(np.abs(xb), axis=1) / 7.0
    scales = np.where(scales == 0, 1.0, scales)
    codes = np.round(xb / scales[:, None])
    codes = np.clip(codes, -7, 7).astype(np.int64)
    offset = (codes + 8).astype(np.uint8).reshape(n)
    low = offset[0::2]
    high = offset[1::2]
    packed = (low | (high << 4)).astype(np.uint8)
    return packed, scales.astype(np.float32)


def _dequant4(packed: np.ndarray, scales: np.ndarray, block_size: int) -> np.ndarray:
    nb = scales.shape[0]
    n = nb * block_size
    low2 = (packed & 0x0F).astype(np.int64) - 8
    high2 = ((packed >> 4) & 0x0F).astype(np.int64) - 8
    unpacked = np.empty(n, dtype=np.int64)
    unpacked[0::2] = low2
    unpacked[1::2] = high2
    xhat = (unpacked.reshape(nb, block_size).astype(np.float64)
            * scales.astype(np.float64)[:, None]).reshape(n)
    return xhat


def _oracle_step(p, grad, m_packed, m_scales, v_packed, v_scales, step, block_size,
                  lr, beta1, beta2, eps, weight_decay) -> dict:
    m_prev = _dequant4(m_packed, m_scales, block_size)
    v_prev = _dequant4(v_packed, v_scales, block_size)

    m = beta1 * m_prev + (1.0 - beta1) * grad
    v = beta2 * v_prev + (1.0 - beta2) * grad * grad

    m_hat = m / (1.0 - beta1 ** step)
    v_hat = v / (1.0 - beta2 ** step)

    update = lr * m_hat / (np.sqrt(v_hat) + eps)
    p_new = p * (1.0 - lr * weight_decay) - update

    m_packed_new, m_scales_new = _quant4(m, block_size)
    v_packed_new, v_scales_new = _quant4(v, block_size)

    return {
        "p_new": p_new, "m": m, "v": v,
        "m_packed": m_packed_new, "m_scales": m_scales_new,
        "v_packed": v_packed_new, "v_scales": v_scales_new,
    }


def _build_case():
    rng = np.random.default_rng(0)
    n = 128
    block_size = 32
    beta1, beta2 = 0.9, 0.999

    p = (rng.standard_normal(n) * 0.02).astype(np.float64)
    m0 = np.zeros(n, dtype=np.float64)
    v0 = np.zeros(n, dtype=np.float64)
    prior_steps = 5
    for _ in range(prior_steps):
        g = (rng.standard_normal(n) * 0.01).astype(np.float64)
        m0 = beta1 * m0 + (1.0 - beta1) * g
        v0 = beta2 * v0 + (1.0 - beta2) * g * g
    m0_packed, m0_scales = _quant4(m0, block_size)
    v0_packed, v0_scales = _quant4(v0, block_size)

    grad = (rng.standard_normal(n) * 0.01).astype(np.float64)
    step = prior_steps + 1
    return dict(p=p, grad=grad, m_packed=m0_packed, m_scales=m0_scales,
                v_packed=v0_packed, v_scales=v0_scales, step=step,
                block_size=block_size, lr=1e-3, beta1=beta1, beta2=beta2,
                eps=1e-8, weight_decay=0.01)


def grade(sol, fx) -> dict:
    case = _build_case()
    oracle = _oracle_step(**case)

    try:
        got = sol.adamw_4bit_step(
            np.array(case["p"], dtype=np.float64), np.array(case["grad"], dtype=np.float64),
            case["m_packed"], case["m_scales"], case["v_packed"], case["v_scales"],
            case["step"], case["block_size"], case["lr"], case["beta1"], case["beta2"],
            case["eps"], case["weight_decay"],
        )
        p_new = np.asarray(got["p_new"], dtype=np.float64)
        m_deq = _dequant4(np.asarray(got["m_packed"]), np.asarray(got["m_scales"]), case["block_size"])
        v_deq = _dequant4(np.asarray(got["v_packed"]), np.asarray(got["v_scales"]), case["block_size"])
    except Exception:
        return {"param_rel_err": float("inf"), "state_rel_err": float("inf")}

    if p_new.shape != oracle["p_new"].shape:
        return {"param_rel_err": float("inf"), "state_rel_err": float("inf")}

    param_rel_err = scorers.rel_err(oracle["p_new"], p_new)

    oracle_m_deq = _dequant4(oracle["m_packed"], oracle["m_scales"], case["block_size"])
    oracle_v_deq = _dequant4(oracle["v_packed"], oracle["v_scales"], case["block_size"])
    oracle_state = np.concatenate([oracle_m_deq, oracle_v_deq])
    got_state = np.concatenate([m_deq, v_deq])
    state_rel_err = scorers.rel_err(oracle_state, got_state)

    return {"param_rel_err": param_rel_err, "state_rel_err": state_rel_err}

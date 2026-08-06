"""Reference oracle generator for evaluation harness."""

import numpy as np


NF4_LEVELS = np.array([
    -1.0, -0.6961928010000001, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791859447956085,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=np.float64)


def get_test_tensors():
    np.random.seed(42)
    t1 = np.random.normal(loc=0.0, scale=1e-4, size=(100, 100))
    t2 = np.random.normal(loc=0.0, scale=1e5, size=(100, 100))
    t3 = np.random.randn(50, 50)
    return {"underflow_heavy": t1, "overflow_heavy": t2, "standard_normal": t3}


def ref_compute_overflow_underflow_fractions(tensor, dtype_str):
    arr = np.asarray(tensor, dtype=np.float64)
    total = arr.size
    if total == 0:
        return {"overflow": 0.0, "underflow": 0.0}

    if dtype_str.lower() in ("fp16", "float16"):
        max_finite = 65504.0
        min_pos_normal = 6.103515625e-5
    elif dtype_str.lower() in ("bf16", "bfloat16"):
        max_finite = 3.3895313892515355e38
        min_pos_normal = 1.1754943508222875e-38
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    abs_arr = np.abs(arr)
    overflow_count = np.sum(abs_arr > max_finite)
    underflow_count = np.sum((abs_arr > 0) & (abs_arr < min_pos_normal))

    return {
        "overflow": float(overflow_count / total),
        "underflow": float(underflow_count / total),
    }


def ref_classify_symptoms(log_entries):
    results = []
    for entry in log_entries:
        grad_norm = float(entry.get("grad_norm", 0.0))
        loss = float(entry.get("loss", 0.0))
        loss_delta = float(entry.get("loss_delta", 0.0))
        unique_act_ratio = float(entry.get("unique_activation_ratio", 1.0))
        is_nan_or_inf = entry.get("is_nan_or_inf", False)

        if is_nan_or_inf or loss != loss or grad_norm > 1e4:
            results.append("FP16_OVERFLOW")
        elif grad_norm == 0.0 and loss > 0.0:
            results.append("FP16_UNDERFLOW")
        elif unique_act_ratio < 0.05:
            results.append("REPRESENTATION_COLLAPSE")
        elif abs(loss_delta) < 1e-6 and grad_norm < 1e-5:
            results.append("GRADIENT_VANISHING")
        else:
            results.append("UNKNOWN")
    return results


def ref_simulate_nf4(tensor, num_cycles=10):
    current = np.asarray(tensor, dtype=np.float64)
    orig = current.copy()

    mse_history = []
    max_err_history = []

    for _ in range(num_cycles):
        abs_max = np.max(np.abs(current))
        if abs_max == 0:
            current = np.zeros_like(current)
        else:
            norm = current / abs_max
            diffs = np.abs(norm[..., None] - NF4_LEVELS)
            idx = np.argmin(diffs, axis=-1)
            current = NF4_LEVELS[idx] * abs_max

        mse = np.mean((orig - current) ** 2)
        max_err = np.max(np.abs(orig - current))
        mse_history.append(mse)
        max_err_history.append(max_err)

    return {
        "final_tensor": current,
        "mse_history": np.array(mse_history, dtype=np.float64),
        "max_err_history": np.array(max_err_history, dtype=np.float64),
    }

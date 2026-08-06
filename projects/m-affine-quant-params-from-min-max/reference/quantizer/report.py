import numpy as np
from quantizer.params import calc_affine_params, quantize, dequantize


def compare_dynamic_vs_static(dataset: list[dict[str, np.ndarray]], static_params: dict[str, tuple[float, int]], qmin: int = 0, qmax: int = 255) -> dict[str, dict[str, float]]:
    tensors = list(dataset[0].keys()) if dataset else []
    report = {}

    for name in tensors:
        dyn_sq_errors = []
        stat_sq_errors = []
        total_q_bytes = 0
        total_f_bytes = 0

        scale_stat, zp_stat = static_params[name]

        for batch in dataset:
            arr = batch[name]
            total_f_bytes += arr.nbytes

            b_min, b_max = float(np.min(arr)), float(np.max(arr))
            scale_dyn, zp_dyn = calc_affine_params(b_min, b_max, qmin, qmax)
            q_dyn = quantize(arr, scale_dyn, zp_dyn, qmin, qmax)
            total_q_bytes += q_dyn.nbytes
            deq_dyn = dequantize(q_dyn, scale_dyn, zp_dyn)
            dyn_sq_errors.append(np.mean((arr - deq_dyn) ** 2))

            q_stat = quantize(arr, scale_stat, zp_stat, qmin, qmax)
            deq_stat = dequantize(q_stat, scale_stat, zp_stat)
            stat_sq_errors.append(np.mean((arr - deq_stat) ** 2))

        dyn_mse = float(np.mean(dyn_sq_errors))
        stat_mse = float(np.mean(stat_sq_errors))
        ratio = float(total_q_bytes / total_f_bytes) if total_f_bytes > 0 else 0.0

        report[name] = {
            "dynamic_mse": dyn_mse,
            "static_mse": stat_mse,
            "size_ratio": ratio,
        }

    return report

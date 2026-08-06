import numpy as np
import ref


def check(workdir):
    try:
        from moe_analyzer.imbalance import compute_imbalance_over_time
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Import error: {e}"}

    fixtures = ref.generate_benchmark_data()
    expected = ref.compute_imbalance_over_time(fixtures["log_entries"])

    try:
        actual = compute_imbalance_over_time(fixtures["log_entries"])
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Execution failed: {e}"}

    exp_arr = np.array(expected["imbalance_ratios"], dtype=np.float64)
    act_arr = np.array(actual.get("imbalance_ratios", []), dtype=np.float64)

    if exp_arr.shape != act_arr.shape:
        return {"rel_err": 1.0, "_note": "Output shape mismatch in imbalance_ratios"}

    norm = np.linalg.norm(exp_arr)
    err = np.linalg.norm(exp_arr - act_arr) / norm if norm > 0 else 0.0

    return {"rel_err": float(err)}

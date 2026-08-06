import numpy as np
import ref


def check(workdir):
    try:
        from moe_analyzer.scaling import fit_and_extrapolate_throughput
        from moe_analyzer.compare import compare_serving_traces
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Import error: {e}"}

    fixtures = ref.generate_benchmark_data()

    exp_scaling = ref.fit_and_extrapolate_throughput(
        fixtures["benchmark_rows"], fixtures["target_ep_degrees"]
    )
    exp_compare = ref.compare_serving_traces(
        fixtures["vllm_trace"], fixtures["sglang_trace"]
    )

    try:
        act_scaling = fit_and_extrapolate_throughput(
            fixtures["benchmark_rows"], fixtures["target_ep_degrees"]
        )
        act_compare = compare_serving_traces(
            fixtures["vllm_trace"], fixtures["sglang_trace"]
        )
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Execution failed: {e}"}

    targets = fixtures["target_ep_degrees"]
    exp_vals = np.array([exp_scaling["extrapolated_throughput"][t] for t in targets], dtype=np.float64)
    act_vals = np.array([act_scaling.get("extrapolated_throughput", {}).get(t, 0.0) for t in targets], dtype=np.float64)

    norm_scaling = np.linalg.norm(exp_vals)
    err_scaling = np.linalg.norm(exp_vals - act_vals) / norm_scaling if norm_scaling > 0 else 0.0

    exp_ratio = float(exp_compare["latency_ratio_vllm_vs_sglang"])
    act_ratio = float(act_compare.get("latency_ratio_vllm_vs_sglang", 0.0))
    err_ratio = abs(exp_ratio - act_ratio) / abs(exp_ratio) if exp_ratio != 0 else 0.0

    total_err = float(err_scaling + err_ratio)
    return {"rel_err": total_err}

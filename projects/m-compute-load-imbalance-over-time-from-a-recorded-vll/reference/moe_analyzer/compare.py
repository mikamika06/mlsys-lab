import numpy as np


def compare_serving_traces(vllm_trace, sglang_trace):
    v_dispatch = np.array([r["dispatch_latency_ms"] for r in vllm_trace], dtype=np.float64)
    v_exec = np.array([r["exec_latency_ms"] for r in vllm_trace], dtype=np.float64)

    s_dispatch = np.array([r["dispatch_latency_ms"] for r in sglang_trace], dtype=np.float64)
    s_exec = np.array([r["exec_latency_ms"] for r in sglang_trace], dtype=np.float64)

    v_total = v_dispatch + v_exec
    s_total = s_dispatch + s_exec

    v_avg_total = float(np.mean(v_total))
    s_avg_total = float(np.mean(s_total))

    latency_ratio = v_avg_total / s_avg_total if s_avg_total > 0 else 1.0
    dispatch_diff_ms = float(np.mean(v_dispatch) - np.mean(s_dispatch))

    return {
        "vllm_avg_latency_ms": v_avg_total,
        "sglang_avg_latency_ms": s_avg_total,
        "latency_ratio_vllm_vs_sglang": float(latency_ratio),
        "dispatch_difference_ms": dispatch_diff_ms,
    }

from quantbench.serving import calculate_model_memory, simulate_decode_throughput
from quantbench.metrics import compute_memory_delta, compute_throughput_ratio


def run_benchmark_suite(configs):
    results = []
    for cfg in configs:
        m_fp16 = calculate_model_memory(cfg, "fp16")
        m_w4 = calculate_model_memory(cfg, "w4a16")
        t_fp16 = simulate_decode_throughput(cfg, "fp16", 4)
        t_w4 = simulate_decode_throughput(cfg, "w4a16", 4)
        results.append({
            "memory_delta": compute_memory_delta(m_fp16, m_w4),
            "throughput_ratio": compute_throughput_ratio(t_fp16, t_w4)
        })
    return results

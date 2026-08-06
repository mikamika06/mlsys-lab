import numpy as np


def run_benchmark_pass(backend: str, batch_size: int, prompt_len: int, gen_len: int) -> dict[str, float]:
    """Simulate serving execution for a backend pass and return performance metrics."""
    base_multiplier = 1.0
    if backend == "FLASH_ATTN":
        base_multiplier = 0.55
    elif backend == "FLASHINFER":
        base_multiplier = 0.75
    elif backend == "XFORMERS":
        base_multiplier = 1.10

    total_tokens = batch_size * (prompt_len + gen_len)
    ttft = (prompt_len * 0.08 * base_multiplier) + (batch_size * 0.02)
    itl = (0.005 * base_multiplier) + (batch_size * 0.0005)
    total_time = ttft + (gen_len * itl)
    throughput = total_tokens / total_time

    return {
        "backend": backend,
        "batch_size": float(batch_size),
        "prompt_len": float(prompt_len),
        "gen_len": float(gen_len),
        "ttft_ms": round(float(ttft * 1000.0), 3),
        "itl_ms": round(float(itl * 1000.0), 3),
        "throughput_tok_s": round(float(throughput), 3),
    }


def generate_tradeoff_report(results: list[dict[str, float]]) -> dict[str, list[dict[str, float]]]:
    """Structure raw benchmark pass metrics into TTFT vs ITL tradeoff report."""
    grouped = {}
    for res in results:
        backend = str(res["backend"])
        if backend not in grouped:
            grouped[backend] = []
        grouped[backend].append(res)

    for backend in grouped:
        grouped[backend].sort(key=lambda x: (x["batch_size"], x["prompt_len"]))

    return {"backends": grouped}

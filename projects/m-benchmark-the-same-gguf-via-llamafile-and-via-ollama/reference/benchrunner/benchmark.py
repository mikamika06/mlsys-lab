def attribute_runner_delta(bench_data: dict) -> dict:
    """Attributes throughput and latency differences between llamafile and Ollama."""
    gen_tokens = bench_data["gen_tokens"]
    lf = bench_data["llamafile"]
    ol = bench_data["ollama"]

    lf_gen_tps = round(gen_tokens / (lf["gen_ms"] / 1000.0), 2)
    ol_gen_tps = round(gen_tokens / (ol["gen_ms"] / 1000.0), 2)
    delta_gen_tps = round(lf_gen_tps - ol_gen_tps, 2)

    ipc_impact_ms = round(ol["ipc_overhead_ms"] - lf["ipc_overhead_ms"], 2)

    lf_isa = set(lf.get("isa_features", []))
    ol_isa = set(ol.get("isa_features", []))

    isa_factor = round(len(lf_isa) / max(1, len(ol_isa)), 2)

    if lf_isa != ol_isa:
        cause = "isa_mismatch"
    elif abs(ol["ipc_overhead_ms"] - lf["ipc_overhead_ms"]) > 15.0:
        cause = "ipc_overhead"
    elif lf.get("threads") != ol.get("threads"):
        cause = "thread_contention"
    else:
        cause = "bandwidth_saturating"

    return {
        "llamafile_gen_tps": lf_gen_tps,
        "ollama_gen_tps": ol_gen_tps,
        "delta_gen_tps": delta_gen_tps,
        "primary_cause": cause,
        "ipc_impact_ms": ipc_impact_ms,
        "isa_impact_factor": isa_factor,
    }

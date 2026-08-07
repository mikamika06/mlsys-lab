def explain_and_fix_speedup(
    uncompiled_latency_ms: float,
    compiled_latency_ms: float,
    fp16_latency_ms: float
) -> dict[str, float | str | bool]:
    has_speedup = compiled_latency_ms < fp16_latency_ms
    overhead_ratio = uncompiled_latency_ms / max(compiled_latency_ms, 1e-9)
    fix_speedup_ratio = fp16_latency_ms / max(compiled_latency_ms, 1e-9)
    return {
        "has_speedup": has_speedup,
        "uncompiled_overhead_ratio": float(overhead_ratio),
        "compiled_speedup_ratio": float(fix_speedup_ratio),
        "primary_cause": "uncompiled_dispatch_and_unpack_overhead",
        "recommended_fix": "torch.compile"
    }

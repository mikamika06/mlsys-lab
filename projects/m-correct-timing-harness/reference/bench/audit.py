"""Audit engine for detecting flawed benchmark timing traces."""
import numpy as np


def audit_benchmark_trace(trace_data):
    """Analyzes execution traces to detect missing syncs, cold caches, and unmeasured async overhead."""
    raw_times = np.array(trace_data.get("raw_times", []), dtype=np.float64)
    has_sync = bool(trace_data.get("has_cuda_sync", False))
    warmup_count = int(trace_data.get("warmup_iters", 0))
    l2_flushed = bool(trace_data.get("l2_flushed", False))
    async_bytes = float(trace_data.get("unmeasured_async_bytes", 0.0))
    bandwidth_gbps = float(trace_data.get("memory_bandwidth_gbps", 1000.0))

    flaws = []
    if not has_sync:
        flaws.append("MISSING_CUDA_SYNC")
    if warmup_count < 5:
        flaws.append("INSUFFICIENT_WARMUP")
    if not l2_flushed:
        flaws.append("UNFLUSHED_L2_CACHE")
    if async_bytes > 0:
        flaws.append("UNMEASURED_ASYNC_MEMCPY")

    corrected = np.copy(raw_times)

    if "INSUFFICIENT_WARMUP" in flaws and len(corrected) > 2:
        corrected = corrected[2:]

    if "UNFLUSHED_L2_CACHE" in flaws:
        corrected = corrected * 1.25

    if "UNMEASURED_ASYNC_MEMCPY" in flaws:
        add_ms = (async_bytes / (bandwidth_gbps * 1e9)) * 1000.0
        corrected = corrected + add_ms

    if "MISSING_CUDA_SYNC" in flaws:
        corrected = corrected * 3.5

    return {
        "flaws_detected": sorted(flaws),
        "is_valid": len(flaws) == 0,
        "corrected_mean_ms": float(np.mean(corrected)),
        "corrected_std_ms": float(np.std(corrected, ddof=1)) if len(corrected) > 1 else 0.0
    }

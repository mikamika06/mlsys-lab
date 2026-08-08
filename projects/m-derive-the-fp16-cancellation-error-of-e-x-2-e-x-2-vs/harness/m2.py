import sys

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from reduction.bandwidth import softmax_memory_traffic, achieved_bandwidth_GBps
    except ImportError:
        sys.path.pop(0)
        return {"_note": "Could not import reduction.bandwidth"}
    sys.path.pop(0)

    out = {"traffic_match": 0.0, "bw_match": 0.0}

    try:
        fused_bytes = softmax_memory_traffic(2048, 2, fused=True)
        unfused_bytes = softmax_memory_traffic(2048, 2, fused=False)
    except NotImplementedError:
        return out

    if fused_bytes == 8192 and unfused_bytes == 24576:
        out["traffic_match"] = 1.0
    else:
        out["_note"] = f"Got {fused_bytes} (fused) and {unfused_bytes} (unfused) traffic bytes. Expected 8192 and 24576."
        return out

    try:
        bw_fused = achieved_bandwidth_GBps(1000000, 2, 1.0, fused=True)
        bw_unfused = achieved_bandwidth_GBps(1000000, 2, 3.0, fused=False)
    except NotImplementedError:
        return out

    # Expected: fused = 4e6 bytes / 1ms = 4.0 GB/s
    # Expected: unfused = 12e6 bytes / 3ms = 4.0 GB/s
    if abs(bw_fused - 4.0) < 1e-4 and abs(bw_unfused - 4.0) < 1e-4:
        out["bw_match"] = 1.0
    else:
        out["_note"] = f"Got GB/s: {bw_fused:.2f} (fused), {bw_unfused:.2f} (unfused). Both should be exactly 4.0."

    return out

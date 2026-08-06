import numpy as np

TRACE_EVENTS = [
    {"name": "AttentionBlock::forward", "ph": "B", "ts": 10},
    {"name": "flash_attn_fwd_kernel", "ph": "X", "cat": "kernel", "ts": 15, "dur": 50},
    {"name": "AttentionBlock::forward", "ph": "E", "ts": 70},
    {"name": "Linear::forward", "ph": "B", "ts": 80},
    {"name": "gemm_kernel", "ph": "X", "cat": "kernel", "ts": 85, "dur": 30},
    {"name": "Linear::forward", "ph": "E", "ts": 120},
]

PROFILES = [
    {
        "op_id": "flash_attn_op",
        "seq_lengths": [512, 1024, 2048, 4096],
        "peak_memory_bytes": [1048576, 2097152, 4194304, 8388608]
    },
    {
        "op_id": "naive_attn_op",
        "seq_lengths": [512, 1024, 2048, 4096],
        "peak_memory_bytes": [2097152, 8388608, 33554432, 134217728]
    }
]

FALLBACK_CONFIGS = [
    ({"dtype": "float16", "head_dim": 128, "is_contiguous": True, "alignment": 16}, "FLASH_ATTENTION_ELIGIBLE"),
    ({"dtype": "float32", "head_dim": 128, "is_contiguous": True, "alignment": 16}, "UNSUPPORTED_DTYPE"),
    ({"dtype": "float16", "head_dim": 48, "is_contiguous": True, "alignment": 16}, "INVALID_HEAD_DIM"),
    ({"dtype": "float16", "head_dim": 128, "is_contiguous": False, "alignment": 16}, "NON_CONTIGUOUS_LAYOUT"),
    ({"dtype": "float16", "head_dim": 128, "is_contiguous": True, "alignment": 4}, "MISALIGNED_ADDRESS")
]

def ref_attribute_kernels(events):
    attributed = []
    stack = []
    for event in sorted(events, key=lambda x: x.get("ts", 0)):
        ev_type = event.get("ph")
        if ev_type == "B":
            stack.append(event)
        elif ev_type == "E":
            if stack:
                stack.pop()
        elif ev_type == "X":
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)
            cat = event.get("cat", "")
            if cat == "kernel":
                parent_scope = stack[-1]["name"] if stack else "root"
                attributed.append({
                    "name": event.get("name"),
                    "scope": parent_scope,
                    "dur": dur,
                    "ts": ts
                })
    return attributed

def ref_analyze_allocations(profiles):
    flagged = []
    for op in profiles:
        seq_lens = np.array(op["seq_lengths"], dtype=np.float64)
        mem_bytes = np.array(op["peak_memory_bytes"], dtype=np.float64)
        log_s = np.log(seq_lens)
        log_m = np.log(mem_bytes)
        poly = np.polyfit(log_s, log_m, 1)
        exponent = poly[0]
        flagged.append({
            "op_id": op["op_id"],
            "exponent": float(exponent),
            "is_quadratic": bool(exponent >= 1.75)
        })
    return flagged

def ref_diagnose_fallback(config):
    dtype = config.get("dtype")
    head_dim = config.get("head_dim")
    is_contiguous = config.get("is_contiguous", True)
    alignment = config.get("alignment", 16)

    if not is_contiguous:
        return "NON_CONTIGUOUS_LAYOUT"
    if dtype not in ("float16", "bfloat16"):
        return "UNSUPPORTED_DTYPE"
    if head_dim not in (32, 64, 128, 256):
        return "INVALID_HEAD_DIM"
    if alignment % 16 != 0:
        return "MISALIGNED_ADDRESS"
    return "FLASH_ATTENTION_ELIGIBLE"

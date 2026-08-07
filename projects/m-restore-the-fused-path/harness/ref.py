import numpy as np

WARNINGS = [
    "WARNING: non-contiguous tensor layout detected; falling back to slow path",
    "WARNING: stride mismatch in key-value cache; disabling fused kernel",
    "WARNING: unsupported head dimension alignment; redirecting to standard attention"
]

CONFIGS = [
    {"layer_id": 0, "shape": (16, 32, 128, 64), "strides": (262144, 8192, 64, 1), "contiguous": True},
    {"layer_id": 1, "shape": (16, 32, 128, 64), "strides": (260000, 8192, 64, 1), "contiguous": False},
    {"layer_id": 2, "shape": (16, 32, 128, 64), "strides": (262144, 8192, 64, 1), "contiguous": True},
]

def triage_warnings(logs):
    out = []
    for log in logs:
        if "non-contiguous" in log:
            out.append("layout")
        elif "stride mismatch" in log:
            out.append("stride")
        elif "head dimension" in log:
            out.append("alignment")
        else:
            out.append("unknown")
    return out

def audit_contiguity(configs):
    results = []
    for cfg in configs:
        shape = cfg["shape"]
        strides = cfg["strides"]
        expected_stride = 1
        is_contig = True
        for dim, stride in zip(reversed(shape), reversed(strides)):
            if stride != expected_stride:
                is_contig = False
                break
            expected_stride *= dim
        results.append({"layer_id": cfg["layer_id"], "contiguous": is_contig and cfg["contiguous"]})
    return results

def restore_path(cfg, warning_log):
    triage = triage_warnings([warning_log])
    audit = audit_contiguity([cfg])
    if not audit[0]["contiguous"] or triage[0] != "unknown":
        return False
    return True

LOG_DATA = """
LOG LINE: [2026-06-01 10:00:01] torch._dynamo: [INFO] Graph break: torch.Tensor size or shape check in forward at line 42 (reason: PyTorch tensor item conversion)
LOG LINE: [2026-06-01 10:00:02] torch._dynamo: [INFO] Graph break: unsupported higher-order control flow at line 88 (reason: explicit Python conditional on tensor value)
LOG LINE: [2026-06-01 10:00:03] torch._dynamo: [INFO] Graph break: torch.Tensor size or shape check in forward at line 42 (reason: PyTorch tensor item conversion)
LOG LINE: [2026-06-01 10:00:04] torch._dynamo: [INFO] Graph break: unsupported global side-effect at line 105 (reason: mutating global state)
"""

PARSED_RESULT = {
    "total_breaks": 4,
    "unique_types": 3,
    "break_counts": {
        "PyTorch tensor item conversion": 2,
        "explicit Python conditional on tensor value": 1,
        "mutating global state": 1,
    }
}

def parse_log(log_text):
    lines = [l.strip() for l in log_text.strip().splitlines() if l.strip()]
    counts = {}
    for line in lines:
        if "reason:" in line:
            reason = line.split("reason:")[1].strip().rstrip(")")
            counts[reason] = counts.get(reason, 0) + 1
    return {
        "total_breaks": len(lines),
        "unique_types": len(counts),
        "break_counts": counts
    }

import re


def parse_server_log(log_text: str) -> dict:
    """Parse mlx_lm.server log to diagnose IOGPUMemory panics and memory metrics."""
    res = {
        "requested_mb": 0,
        "wired_limit_mb": 0,
        "has_panic": False,
        "panic_reason": "NONE"
    }

    req_match = re.search(r"alloc_request_mb=(\d+)", log_text)
    if req_match:
        res["requested_mb"] = int(req_match.group(1))

    limit_match = re.search(r"iogpu\.wired_mem_limit_mb\s*=\s*(\d+)", log_text)
    if limit_match:
        res["wired_limit_mb"] = int(limit_match.group(1))

    if "IOGPUMemory" in log_text or "kernel panic" in log_text.lower():
        res["has_panic"] = True
        if "exceeded wired limit" in log_text.lower():
            res["panic_reason"] = "WIRED_LIMIT_EXCEEDED"
        elif "allocation failed" in log_text.lower():
            res["panic_reason"] = "ALLOCATION_FAILED"
        else:
            res["panic_reason"] = "UNKNOWN_PANIC"

    return res

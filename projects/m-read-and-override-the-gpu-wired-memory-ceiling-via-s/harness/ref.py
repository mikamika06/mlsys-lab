def predict_wired_limit_mb(memsize_bytes: int) -> int:
    mem_gb = memsize_bytes / (1024 ** 3)
    if mem_gb <= 16:
        ratio = 0.65
    elif mem_gb <= 32:
        ratio = 0.70
    elif mem_gb <= 64:
        ratio = 0.75
    else:
        ratio = 0.80
    total_mb = memsize_bytes // (1024 * 1024)
    return int(total_mb * ratio)


def generate_sysctl_override(memsize_bytes: int, target_percentage: float) -> str:
    if not (50.0 <= target_percentage <= 95.0):
        raise ValueError("Target percentage must be between 50.0 and 95.0")
    total_mb = memsize_bytes // (1024 * 1024)
    target_mb = int(total_mb * (target_percentage / 100.0))
    return f"sudo sysctl iogpu.wired_mem_limit_mb={target_mb}"


def parse_server_log(log_text: str) -> dict:
    import re
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


SAMPLE_MEMSIZES = [
    16 * 1024 * 1024 * 1024,
    32 * 1024 * 1024 * 1024,
    64 * 1024 * 1024 * 1024,
    128 * 1024 * 1024 * 1024,
]

SAMPLE_LOGS = [
    """[2026-03-10 10:15:22] INFO mlx_lm.server: Initializing server
[2026-03-10 10:15:22] INFO sysctl setting: iogpu.wired_mem_limit_mb = 49152
[2026-03-10 10:15:30] INFO alloc_request_mb=12288
[2026-03-10 10:15:35] INFO alloc_request_mb=24576
""",
    """[2026-03-10 10:20:00] INFO mlx_lm.server: Initializing server
[2026-03-10 10:20:00] INFO sysctl setting: iogpu.wired_mem_limit_mb = 24576
[2026-03-10 10:20:05] ERROR alloc_request_mb=30720
[2026-03-10 10:20:05] CRITICAL Kernel Panic: IOGPUMemory allocation failure - exceeded wired limit
""",
    """[2026-03-10 10:35:10] INFO mlx_lm.server: Running request batch
[2026-03-10 10:35:10] INFO sysctl setting: iogpu.wired_mem_limit_mb = 98304
[2026-03-10 10:35:12] ERROR alloc_request_mb=100000
[2026-03-10 10:35:12] CRITICAL IOGPUMemory allocation failed unexpectedly
"""
]

import re


def parse_oom_log(log_text):
    """Parses mlx_lm.server stderr log text for OOM crash telemetry."""
    requested_mb = 0.0
    limit_mb = 0.0
    active_tokens = 0
    batch_size = 0

    req_match = re.search(r"MemoryLimitError:\s*\[metal::malloc\]\s*Cannot allocate\s+([0-9.]+)\s*MB", log_text)
    if req_match:
        requested_mb = float(req_match.group(1))

    limit_match = re.search(r"exceeds\s+device\040limit\s+([0-9.]+)\s*MB", log_text)
    if not limit_match:
        limit_match = re.search(r"limit\s+([0-9.]+)\s*MB", log_text)
    if limit_match:
        limit_mb = float(limit_match.group(1))

    tokens_match = re.search(r"active_tokens\s*=\s*(\d+)", log_text)
    if tokens_match:
        active_tokens = int(tokens_match.group(1))

    batch_match = re.search(r"batch_size\s*=\s*(\d+)", log_text)
    if batch_match:
        batch_size = int(batch_match.group(1))

    peak_mb = limit_mb + requested_mb

    return {
        "requested_mb": requested_mb,
        "limit_mb": limit_mb,
        "peak_mb": peak_mb,
        "active_tokens": active_tokens,
        "batch_size": batch_size,
        "is_oom": requested_mb > 0.0 or "MemoryLimitError" in log_text,
    }

import re

def parse_trt_llm_log(log_content: str) -> dict:
    stats = {"layers": 0, "hidden_units": 0, "draft_tokens": 0, "peak_memory_mb": 0.0}
    for line in log_content.splitlines():
        if "DraftEngine" in line or "draft" in line.lower():
            m = re.search(r"layers[=:]\s*(\d+)", line, re.IGNORECASE)
            if m:
                stats["layers"] = int(m.group(1))
            m = re.search(r"hidden[=:]\s*(\d+)", line, re.IGNORECASE)
            if m:
                stats["hidden_units"] = int(m.group(1))
            m = re.search(r"spec_tokens[=:]\s*(\d+)", line, re.IGNORECASE)
            if m:
                stats["draft_tokens"] = int(m.group(1))
        if "Peak memory" in line or "peak_memory" in line.lower():
            m = re.search(r"([\d\.]+)\s*MB", line, re.IGNORECASE)
            if m:
                stats["peak_memory_mb"] = float(m.group(1))
    return stats

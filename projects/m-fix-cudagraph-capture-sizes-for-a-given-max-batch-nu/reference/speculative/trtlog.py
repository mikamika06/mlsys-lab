import re


def extract_draft_engine_stats(log_content: str) -> dict:
    stats = {}
    m_latency = re.search(r"Draft engine avg latency:\s*([0-9.]+)\s*ms", log_content)
    if m_latency:
        stats["avg_latency_ms"] = float(m_latency.group(1))
    else:
        stats["avg_latency_ms"] = 0.0

    m_accept = re.search(r"Draft acceptance rate:\s*([0-9.]+)", log_content)
    if m_accept:
        stats["acceptance_rate"] = float(m_accept.group(1))
    else:
        stats["acceptance_rate"] = 0.0

    m_memory = re.search(r"Draft engine peak memory:\s*([0-9]+)\s*bytes", log_content)
    if m_memory:
        stats["peak_memory_bytes"] = int(m_memory.group(1))
    else:
        stats["peak_memory_bytes"] = 0

    return stats

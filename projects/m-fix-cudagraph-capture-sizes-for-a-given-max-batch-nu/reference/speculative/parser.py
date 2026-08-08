import re


def parse_draft_stats(log_content: str) -> dict:
    stats = {}
    for line in log_content.splitlines():
        if "DRAFT_ENGINE_LATENCY:" in line:
            match = re.search(r"DRAFT_ENGINE_LATENCY:\s*([0-9.]+)", line)
            if match:
                stats["latency_ms"] = float(match.group(1))
        elif "DRAFT_ENGINE_ACCEPTED:" in line:
            match = re.search(r"DRAFT_ENGINE_ACCEPTED:\s*([0-9.]+)", line)
            if match:
                stats["accepted_tokens"] = float(match.group(1))
        elif "DRAFT_ENGINE_THROUGHPUT:" in line:
            match = re.search(r"DRAFT_ENGINE_THROUGHPUT:\s*([0-9.]+)", line)
            if match:
                stats["throughput"] = float(match.group(1))
    return stats

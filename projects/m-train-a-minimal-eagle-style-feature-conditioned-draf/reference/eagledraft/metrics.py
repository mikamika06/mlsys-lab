import re
from typing import List, Dict


def parse_vllm_eagle_log(log_lines: List[str]) -> Dict[str, float]:
    accepted_counts = []

    pattern = re.compile(r"accepted_tokens:\s*(\d+)")

    for line in log_lines:
        if "EAGLE" in line or "speculative" in line.lower():
            match = pattern.search(line)
            if match:
                accepted_counts.append(int(match.group(1)))

    if not accepted_counts:
        return {"mean_accepted_length": 0.0, "total_steps": 0.0}

    mean_accepted = float(sum(accepted_counts) / len(accepted_counts))
    return {
        "mean_accepted_length": mean_accepted,
        "total_steps": float(len(accepted_counts)),
    }

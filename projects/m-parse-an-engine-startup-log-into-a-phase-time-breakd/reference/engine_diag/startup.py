import re
from typing import Dict


def parse_startup_log(log_text: str) -> Dict[str, float]:
    """Parse engine startup log into phase durations."""
    phase_times = {}
    pattern = re.compile(r"\[PHASE:([a-zA-Z0-9_]+)\]\s+START=(\d+\.?\d*)\s+END=(\d+\.?\d*)")
    for line in log_text.strip().splitlines():
        match = pattern.search(line)
        if match:
            phase_name, start_s, end_s = match.groups()
            duration = float(end_s) - float(start_s)
            phase_times[phase_name] = round(duration, 4)
    return phase_times


def total_startup_time(phase_breakdown: Dict[str, float]) -> float:
    """Calculate total engine startup time."""
    return round(sum(phase_breakdown.values()), 4)

"""Telemetry counter analysis."""


def detect_runner_reloads(request_counters):
    """Detect request indices where a runner reload occurred from telemetry counters."""
    reloads = []
    if not request_counters:
        return reloads

    prev_runner_id = None
    prev_total_loads = None

    for i, item in enumerate(request_counters):
        curr_runner_id = item.get("runner_id")
        curr_total_loads = item.get("total_loads", 0)

        if i > 0:
            if curr_runner_id != prev_runner_id or curr_total_loads > prev_total_loads:
                reloads.append(i)

        prev_runner_id = curr_runner_id
        prev_total_loads = curr_total_loads

    return reloads

def select_sessions_to_offload(sessions, gpu_budget):
    sorted_sessions = sorted(sessions, key=lambda s: (s.get("priority", 0), s.get("last_accessed", 0)))
    offloaded = []
    current_usage = sum(s["tokens"] for s in sessions)
    for s in sorted_sessions:
        if current_usage <= gpu_budget:
            break
        offloaded.append(s["id"])
        current_usage -= s["tokens"]
    return offloaded

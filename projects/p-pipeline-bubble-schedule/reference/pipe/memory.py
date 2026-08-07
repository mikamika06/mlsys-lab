def track_activation_memory(schedule_events: list, num_stages: int) -> list:
    if not schedule_events:
        return [0] * num_stages

    stage_events = {p: [] for p in range(num_stages)}
    for ev in schedule_events:
        p = ev["stage"]
        stage_events[p].append(ev)

    peak_memory = [0] * num_stages

    for p in range(num_stages):
        current_mem = 0
        peak_mem = 0
        events_p = stage_events[p]

        timeline = []
        for ev in events_p:
            if ev["type"] == "F":
                timeline.append((ev["start"] + ev["duration"], 1))
            elif ev["type"] in ("B", "B_input"):
                timeline.append((ev["start"], -1))

        timeline.sort(key=lambda x: (x[0], x[1]))

        for _, delta in timeline:
            current_mem += delta
            if current_mem > peak_mem:
                peak_mem = current_mem
        peak_memory[p] = peak_mem

    return peak_memory

def is_within_memory_budget(schedule_events: list, num_stages: int, max_units: int) -> bool:
    peaks = track_activation_memory(schedule_events, num_stages)
    return all(p <= max_units for p in peaks)

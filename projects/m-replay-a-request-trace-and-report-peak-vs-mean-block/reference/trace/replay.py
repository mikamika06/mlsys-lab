def replay_trace(trace):
    max_t = 0
    events = []
    for req in trace:
        start = req["start"]
        duration = req["duration"]
        blocks = req["blocks"]
        end = start + duration
        max_t = max(max_t, end)
        events.append((start, end, blocks))

    timeline = [0] * (max_t + 1)
    for start, end, blocks in events:
        for t in range(start, end):
            if t < len(timeline):
                timeline[t] += blocks
    return timeline

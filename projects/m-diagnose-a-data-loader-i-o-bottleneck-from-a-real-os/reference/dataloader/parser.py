def parse_nvtx_timeline(events):
    stack = []
    timeline = []
    for ts, kind, name in sorted(events, key=lambda x: x[0]):
        if kind == "push":
            stack.append((ts, name))
            timeline.append((ts, len(stack), name))
        elif kind == "pop":
            if stack:
                start_ts, popped_name = stack.pop()
                timeline.append((ts, len(stack) + 1, popped_name))
    return timeline

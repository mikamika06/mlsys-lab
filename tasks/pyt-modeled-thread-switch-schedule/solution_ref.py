def gil_schedule(interval, streams):
    streams = [list(s) for s in streams]
    pos = [0] * len(streams)
    total = 0
    current = 0
    result = []

    while any(pos[i] < len(streams[i]) for i in range(len(streams))):
        if pos[current] >= len(streams[current]):
            current = (current + 1) % len(streams)
            continue

        total += streams[current][pos[current]]
        pos[current] += 1
        result.append(current)

        if total >= interval:
            total = 0
            for step in range(1, len(streams) + 1):
                nxt = (current + step) % len(streams)
                if pos[nxt] < len(streams[nxt]):
                    current = nxt
                    break

    return result

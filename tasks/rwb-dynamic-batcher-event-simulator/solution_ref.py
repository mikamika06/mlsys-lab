def dynamic_batcher_simulate(arrivals, preferred_batch_size, max_queue_delay):
    arrivals = sorted(int(a) for a in arrivals)
    n = len(arrivals)
    i = 0
    queue = []
    times = []
    sizes = []

    while i < n or queue:
        next_arrival = arrivals[i] if i < n else None
        next_timeout = queue[0] + max_queue_delay if queue else None

        if next_arrival is not None and (next_timeout is None or next_arrival <= next_timeout):
            t = next_arrival
            while i < n and arrivals[i] == t:
                queue.append(arrivals[i])
                i += 1
                if len(queue) == preferred_batch_size:
                    times.append(t)
                    sizes.append(preferred_batch_size)
                    queue = []
        else:
            t = next_timeout
            times.append(t)
            sizes.append(len(queue))
            queue = []

    return times, sizes

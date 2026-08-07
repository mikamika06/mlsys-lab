def fraction_exposed(trace, idle_timeout):
    if not trace:
        return 0.0
    sorted_trace = sorted(trace, key=lambda x: x["arrival"])
    exposed = 0
    last_time = -float("inf")
    for req in sorted_trace:
        arrival = req["arrival"]
        if arrival - last_time > idle_timeout:
            exposed += 1
        last_time = arrival
    return float(exposed) / float(len(sorted_trace))

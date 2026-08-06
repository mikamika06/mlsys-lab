def simulate_ttl(ttl, events):
    states = []
    last_req = None
    for t in events:
        if last_req is None:
            states.append("loaded")
            last_req = t
        else:
            if (t - last_req) <= ttl:
                states.append("loaded")
                last_req = t
            else:
                states.append("evicted")
                last_req = t
    return states

import math


def simulate_peak_blocks(events, block_size):
    """Simulate arrival events and calculate peak block usage over time."""
    timeline = trace_block_timeline(events, block_size)
    if not timeline:
        return 0
    return max(blocks for _, blocks in timeline)


def trace_block_timeline(events, block_size):
    """Generate a step-by-step timeline of active block usage."""
    discrete_events = []
    for req in events:
        arrival_time = req["arrival_time"]
        prompt_len = req["prompt_len"]
        gen_len = req["gen_len"]
        decode_speed = req.get("decode_speed", 1)

        total_len = prompt_len + gen_len
        end_time = arrival_time + (gen_len * decode_speed)

        for step in range(gen_len + 1):
            t = arrival_time + (step * decode_speed)
            curr_len = prompt_len + step
            blocks = math.ceil(curr_len / block_size) if curr_len > 0 else 0
            discrete_events.append((t, req["req_id"], blocks))

        discrete_events.append((end_time + 1e-9, req["req_id"], 0))

    discrete_events.sort(key=lambda x: x[0])

    active_reqs = {}
    timeline = []

    for t, req_id, blocks in discrete_events:
        if blocks == 0:
            active_reqs.pop(req_id, None)
        else:
            active_reqs[req_id] = blocks

        total_blocks = sum(active_reqs.values())
        timeline.append((t, total_blocks))

    return timeline

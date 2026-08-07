import math


def simulate_peak_blocks(events, block_size):
    timeline = trace_block_timeline(events, block_size)
    if not timeline:
        return 0
    return max(blocks for _, blocks in timeline)


def trace_block_timeline(events, block_size):
    discrete_events = []
    for req in events:
        arrival_time = req["arrival_time"]
        prompt_len = req["prompt_len"]
        gen_len = req["gen_len"]
        decode_speed = req.get("decode_speed", 1)

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


def compute_paged_waste(length_histogram, block_size):
    total_waste = 0
    for length, count in length_histogram.items():
        if length == 0:
            continue
        allocated_tokens = math.ceil(length / block_size) * block_size
        waste_per_seq = allocated_tokens - length
        total_waste += waste_per_seq * count
    return total_waste


def compute_contiguous_waste(length_histogram, max_possible_len):
    total_waste = 0
    for length, count in length_histogram.items():
        waste_per_seq = max_possible_len - length
        total_waste += waste_per_seq * count
    return total_waste


def compute_waste_ratio(length_histogram, block_size, max_possible_len):
    paged = compute_paged_waste(length_histogram, block_size)
    contiguous = compute_contiguous_waste(length_histogram, max_possible_len)
    if paged == 0:
        return float("inf") if contiguous > 0 else 1.0
    return contiguous / paged


TRACES = [
    (
        [
            {"req_id": 101, "arrival_time": 0, "prompt_len": 12, "gen_len": 8, "decode_speed": 1},
            {"req_id": 102, "arrival_time": 3, "prompt_len": 30, "gen_len": 10, "decode_speed": 1},
            {"req_id": 103, "arrival_time": 5, "prompt_len": 5, "gen_len": 15, "decode_speed": 2},
        ],
        16,
    ),
    (
        [
            {"req_id": 1, "arrival_time": 1, "prompt_len": 64, "gen_len": 32, "decode_speed": 1},
            {"req_id": 2, "arrival_time": 1, "prompt_len": 128, "gen_len": 16, "decode_speed": 1},
        ],
        32,
    ),
    (
        [
            {"req_id": 201, "arrival_time": 0, "prompt_len": 1, "gen_len": 1, "decode_speed": 1},
            {"req_id": 202, "arrival_time": 1, "prompt_len": 1, "gen_len": 1, "decode_speed": 1},
            {"req_id": 203, "arrival_time": 2, "prompt_len": 100, "gen_len": 50, "decode_speed": 1},
        ],
        16,
    ),
    (
        [
            {"req_id": 301, "arrival_time": 10, "prompt_len": 500, "gen_len": 100, "decode_speed": 1},
            {"req_id": 302, "arrival_time": 12, "prompt_len": 200, "gen_len": 50, "decode_speed": 1},
            {"req_id": 303, "arrival_time": 15, "prompt_len": 300, "gen_len": 80, "decode_speed": 1},
        ],
        64,
    ),
    (
        [
            {"req_id": 401, "arrival_time": 0, "prompt_len": 15, "gen_len": 1, "decode_speed": 1},
            {"req_id": 402, "arrival_time": 1, "prompt_len": 16, "gen_len": 1, "decode_speed": 1},
            {"req_id": 403, "arrival_time": 2, "prompt_len": 17, "gen_len": 1, "decode_speed": 1},
        ],
        16,
    ),
]

HISTOGRAMS = [
    ({10: 20, 15: 10, 32: 5, 60: 12, 120: 3}, 16, 2048),
    ({1: 100, 16: 50, 17: 25, 256: 10}, 16, 1024),
    ({512: 5, 1024: 5, 1500: 2}, 64, 4096),
]

HISTOGRAMS = [
    {32: 10, 64: 20, 128: 5},
    {15: 4, 31: 8, 63: 2},
    {100: 5, 200: 10, 300: 15}
]

CANDIDATES = [8, 16, 32, 64]

TRACES = [
    [
        {"type": "allocate", "seq_id": 1, "block_id": 10},
        {"type": "allocate", "seq_id": 1, "block_id": 11},
        {"type": "free", "seq_id": 1, "block_id": 10},
        {"type": "free", "seq_id": 1, "block_id": 11},
        {"type": "terminate", "seq_id": 1}
    ],
    [
        {"type": "allocate", "seq_id": 2, "block_id": 20},
        {"type": "allocate", "seq_id": 2, "block_id": 20},
        {"type": "terminate", "seq_id": 2}
    ],
    [
        {"type": "allocate", "seq_id": 3, "block_id": 30},
        {"type": "free", "seq_id": 3, "block_id": 99},
        {"type": "terminate", "seq_id": 3}
    ],
    [
        {"type": "allocate", "seq_id": 4, "block_id": 40},
        {"type": "terminate", "seq_id": 4}
    ]
]


def expected_wasted_tokens(length_histogram, block_size):
    total_seqs = sum(length_histogram.values())
    if total_seqs == 0:
        return 0.0
    total_wasted = 0
    for length, count in length_histogram.items():
        rem = length % block_size
        wasted = 0 if rem == 0 else block_size - rem
        total_wasted += wasted * count
    return float(total_wasted) / float(total_seqs)


def optimal_block_size(length_histogram, candidate_block_sizes, bytes_per_token):
    best_size = candidate_block_sizes[0]
    best_cost = float("inf")
    for b in candidate_block_sizes:
        w = expected_wasted_tokens(length_histogram, b)
        avg_len = sum(l * c for l, c in length_histogram.items()) / sum(length_histogram.values())
        cost = (avg_len + w) * bytes_per_token
        if cost < best_cost:
            best_cost = cost
            best_size = b
    return best_size


def audit_block_trace(trace):
    allocated = set()
    active_per_seq = {}
    double_free = 0
    use_after_free = 0
    leaked = 0

    for event in trace:
        etype = event["type"]
        seq_id = event.get("seq_id")
        block_id = event.get("block_id")

        if etype == "allocate":
            if block_id in allocated:
                double_free += 1
            allocated.add(block_id)
            active_per_seq.setdefault(seq_id, set()).add(block_id)
        elif etype == "free":
            if block_id not in allocated:
                use_after_free += 1
            else:
                allocated.remove(block_id)
                if seq_id in active_per_seq and block_id in active_per_seq[seq_id]:
                    active_per_seq[seq_id].remove(block_id)
        elif etype == "terminate":
            if seq_id in active_per_seq:
                leaked += len(active_per_seq[seq_id])
                for b in active_per_seq[seq_id]:
                    if b in allocated:
                        allocated.remove(b)
                del active_per_seq[seq_id]

    leaked += len(allocated)
    return {
        "double_free": double_free,
        "use_after_free": use_after_free,
        "leaked": leaked,
    }

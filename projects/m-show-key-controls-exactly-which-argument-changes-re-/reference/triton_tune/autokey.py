def check_key_triggers(key_args, call_sequences):
    triggered = []
    last_seen_signature = None
    for seq in call_sequences:
        current_signature = tuple(seq[k] for k in key_args)
        if current_signature != last_seen_signature:
            triggered.append(True)
            last_seen_signature = current_signature
        else:
            triggered.append(False)
    return triggered


def find_true_argmin(sweep_records):
    best_config = None
    min_time = float("inf")
    for record in sweep_records:
        t = record["latency"]
        if t < min_time:
            min_time = t
            best_config = record["config"]
    return best_config


def measure_search_overhead(search_times, hardcoded_time):
    total_search_time = sum(search_times)
    overhead_ratio = total_search_time / max(hardcoded_time, 1e-9)
    return overhead_ratio

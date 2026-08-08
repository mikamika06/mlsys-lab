from draftquant.latency import estimate_draft_latency


def calculate_throughput_ratio(draft_size, target_latency, hw_config, acceptance_rate=0.7, gamma=5):
    t_draft_fp16 = estimate_draft_latency(draft_size, "fp16", hw_config)
    t_draft_int8 = estimate_draft_latency(draft_size, "int8", hw_config)

    if acceptance_rate > 0:
        expected_accepted = (1.0 - (1.0 - acceptance_rate) ** (gamma + 1)) / acceptance_rate
    else:
        expected_accepted = 1.0

    total_time_fp16 = gamma * t_draft_fp16 + target_latency
    total_time_int8 = gamma * t_draft_int8 + target_latency

    tp_fp16 = expected_accepted / total_time_fp16
    tp_int8 = expected_accepted / total_time_int8

    if tp_int8 == 0:
        return 0.0
    return tp_fp16 / tp_int8


def find_int8_crossover_size(param_candidates, target_latency, hw_config, acceptance_rate=0.7, gamma=5):
    sorted_candidates = sorted(param_candidates)
    crossover = None

    for size in sorted_candidates:
        ratio = calculate_throughput_ratio(size, target_latency, hw_config, acceptance_rate, gamma)
        if ratio >= 1.0:
            crossover = size
            break

    return crossover

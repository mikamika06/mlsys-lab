def is_net_end_to_end_win(latency_ratio, alpha_fp16, alpha_int8, target_latency, draft_fp16_latency, gamma=4):
    accepted_tokens_fp16 = alpha_fp16 * gamma
    accepted_tokens_int8 = alpha_int8 * gamma

    tokens_generated_fp16 = 1.0 + accepted_tokens_fp16
    tokens_generated_int8 = 1.0 + accepted_tokens_int8

    step_latency_fp16 = (gamma * draft_fp16_latency) + target_latency
    step_latency_int8 = (gamma * draft_fp16_latency * latency_ratio) + target_latency

    throughput_fp16 = tokens_generated_fp16 / step_latency_fp16
    throughput_int8 = tokens_generated_int8 / step_latency_int8

    return throughput_int8 > throughput_fp16

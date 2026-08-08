def expected_tokens(alpha, K):
    return sum(alpha**i for i in range(K + 1))

def calculate_throughput(s_draft, is_int8, s_target, K, mem_bw, alpha, overheads):
    bytes_per_param = 1 if is_int8 else 2
    draft_overhead = overheads["draft_int8"] if is_int8 else overheads["draft_fp16"]
    t_gen_draft = (s_draft * bytes_per_param) / mem_bw + draft_overhead

    t_verify = (s_target * 2) / mem_bw + K * overheads["target_verify_per_token"] + overheads["target_verify_base"]

    t_step = K * t_gen_draft + t_verify
    e_tokens = expected_tokens(alpha, K)

    return e_tokens / t_step

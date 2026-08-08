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

def find_int8_cutoff(draft_sizes, s_target, K, mem_bw, alphas_fp16, alphas_int8, overheads):
    for s in sorted(draft_sizes):
        tp_fp16 = calculate_throughput(s, False, s_target, K, mem_bw, alphas_fp16[s], overheads)
        tp_int8 = calculate_throughput(s, True, s_target, K, mem_bw, alphas_int8[s], overheads)
        if tp_int8 > tp_fp16:
            return s
    return None

SCENARIOS = [
    {
        "draft_sizes": [100_000_000, 250_000_000, 500_000_000, 1_000_000_000],
        "s_target": 7_000_000_000,
        "K": 4,
        "mem_bw": 300_000_000_000.0,
        "alphas_fp16": {100_000_000: 0.6, 250_000_000: 0.7, 500_000_000: 0.8, 1_000_000_000: 0.85},
        "alphas_int8": {100_000_000: 0.58, 250_000_000: 0.68, 500_000_000: 0.78, 1_000_000_000: 0.83},
        "overheads": {
            "draft_fp16": 0.001,
            "draft_int8": 0.0015,
            "target_verify_base": 0.005,
            "target_verify_per_token": 0.0001
        }
    },
    {
        "draft_sizes": [50_000_000, 100_000_000],
        "s_target": 13_000_000_000,
        "K": 5,
        "mem_bw": 100_000_000_000.0,
        "alphas_fp16": {50_000_000: 0.4, 100_000_000: 0.5},
        "alphas_int8": {50_000_000: 0.35, 100_000_000: 0.45},
        "overheads": {
            "draft_fp16": 0.002,
            "draft_int8": 0.004,
            "target_verify_base": 0.010,
            "target_verify_per_token": 0.0002
        }
    },
    {
        "draft_sizes": [300_000_000, 600_000_000],
        "s_target": 7_000_000_000,
        "K": 3,
        "mem_bw": 800_000_000_000.0,
        "alphas_fp16": {300_000_000: 0.75, 600_000_000: 0.82},
        "alphas_int8": {300_000_000: 0.74, 600_000_000: 0.81},
        "overheads": {
            "draft_fp16": 0.0005,
            "draft_int8": 0.0006,
            "target_verify_base": 0.002,
            "target_verify_per_token": 0.00005
        }
    }
]

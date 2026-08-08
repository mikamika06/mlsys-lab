from speculative_quant.throughput import calculate_throughput

def find_int8_cutoff(draft_sizes, s_target, K, mem_bw, alphas_fp16, alphas_int8, overheads):
    for s in sorted(draft_sizes):
        tp_fp16 = calculate_throughput(s, False, s_target, K, mem_bw, alphas_fp16[s], overheads)
        tp_int8 = calculate_throughput(s, True, s_target, K, mem_bw, alphas_int8[s], overheads)
        if tp_int8 > tp_fp16:
            return s
    return None

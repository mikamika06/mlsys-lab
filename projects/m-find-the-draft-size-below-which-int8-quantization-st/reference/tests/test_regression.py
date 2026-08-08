import sys
sys.path.insert(0, ".")

from speculative_quant.throughput import calculate_throughput
from speculative_quant.cutoff import find_int8_cutoff

def test_cutoff_improves_throughput():
    draft_sizes = [50_000_000, 100_000_000, 500_000_000]
    s_target = 7_000_000_000
    K = 4
    mem_bw = 300_000_000_000.0

    alphas_fp16 = {50_000_000: 0.4, 100_000_000: 0.8, 500_000_000: 0.9}
    alphas_int8 = {50_000_000: 0.3, 100_000_000: 0.5, 500_000_000: 0.88}

    overheads = {
        "draft_fp16": 0.001,
        "draft_int8": 0.0011,
        "target_verify_base": 0.005,
        "target_verify_per_token": 0.0001
    }

    cutoff = find_int8_cutoff(
        draft_sizes, s_target, K, mem_bw,
        alphas_fp16, alphas_int8, overheads
    )

    if cutoff is not None:
        tp_fp16 = calculate_throughput(cutoff, False, s_target, K, mem_bw, alphas_fp16[cutoff], overheads)
        tp_int8 = calculate_throughput(cutoff, True, s_target, K, mem_bw, alphas_int8[cutoff], overheads)
        assert tp_int8 > tp_fp16, f"Cutoff selected {cutoff} but INT8 throughput {tp_int8} <= FP16 throughput {tp_fp16}"

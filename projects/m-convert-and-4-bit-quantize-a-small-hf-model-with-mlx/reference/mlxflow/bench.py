def compute_speedup(runs_fp16, runs_quant):
    if not runs_fp16 or not runs_quant:
        return 0.0
    avg_fp16 = sum(runs_fp16) / len(runs_fp16)
    avg_quant = sum(runs_quant) / len(runs_quant)
    if avg_fp16 == 0.0:
        return 0.0
    return round(avg_quant / avg_fp16, 4)

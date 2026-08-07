def compute_ratios(records):
    out = []
    for r in records:
        ratio = r["FP8_TFLOPS"] / r["FP16_CUBLAS_TFLOPS"]
        out.append({"M": r["M"], "N": r["N"], "K": r["K"], "throughput_ratio": ratio})
    return out

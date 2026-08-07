RAW_LOGS = [
    "SHAPE M=1024 N=1024 K=1024 FP8_TFLOPS=150.5 FP16_CUBLAS_TFLOPS=100.0",
    "SHAPE M=2048 N=2048 K=2048 FP8_TFLOPS=320.0 FP16_CUBLAS_TFLOPS=200.0",
    "SHAPE M=4096 N=4096 K=4096 FP8_TFLOPS=640.0 FP16_CUBLAS_TFLOPS=400.0",
    "SHAPE M=8192 N=8192 K=8192 FP8_TFLOPS=1200.0 FP16_CUBLAS_TFLOPS=800.0",
]


def parse_logs(lines):
    res = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        d = {}
        for p in parts[1:]:
            k, v = p.split("=")
            d[k] = float(v) if "." in v or "E" in v else int(v)
        res.append(d)
    return res


def compute_ratios(records):
    out = []
    for r in records:
        ratio = r["FP8_TFLOPS"] / r["FP16_CUBLAS_TFLOPS"]
        out.append({"M": r["M"], "N": r["N"], "K": r["K"], "throughput_ratio": ratio})
    return out

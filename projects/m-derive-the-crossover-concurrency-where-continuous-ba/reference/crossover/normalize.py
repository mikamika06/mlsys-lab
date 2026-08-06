def normalize_benchmarks(benchmarks):
    mult = {"FP32": 0.5, "FP16": 1.0, "INT8": 1.5, "INT4": 2.0}
    res = []
    for b in benchmarks:
        p = b["precision"]
        factor = mult.get(p, 1.0)
        norm_tps = b["raw_tps"] * factor
        per_gpu = norm_tps / b["gpus"]
        res.append(per_gpu)
    return res

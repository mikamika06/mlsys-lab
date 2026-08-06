import random

def get_test_cases():
    random.seed(42)
    cases = []
    for i in range(5):
        t1 = 0.02 + i * 0.005
        tb = 0.05 + i * 0.01
        tp = 0.01 + i * 0.002
        prompt_len = 128
        gen_len = 64
        cases.append({
            "t1": t1,
            "tb": tb,
            "tp": tp,
            "prompt_len": prompt_len,
            "gen_len": gen_len
        })
    return cases

def compute_crossover(case):
    t1 = case["t1"]
    tb = case["tb"]
    tp = case["tp"]
    pl = case["prompt_len"]
    gl = case["gen_len"]

    for c in range(1, 1000):
        time_batch1 = c * (pl * t1 + gl * t1)
        num_batches = (c + 31) // 32
        time_cb = num_batches * tb + (pl + gl) * tp * num_batches
        if time_cb < time_batch1:
            return c
    return 1000

def get_benchmarks():
    return [
        {"name": "b1", "raw_tps": 100.0, "gpus": 1, "precision": "FP16", "warmup": True, "requests": 50},
        {"name": "b2", "raw_tps": 180.0, "gpus": 2, "precision": "INT8", "warmup": True, "requests": 100},
        {"name": "b3", "raw_tps": 40.0, "gpus": 1, "precision": "FP32", "warmup": False, "requests": 1},
        {"name": "b4", "raw_tps": 220.0, "gpus": 4, "precision": "INT4", "warmup": True, "requests": 200},
        {"name": "b5", "raw_tps": 90.0, "gpus": 1, "precision": "FP16", "warmup": True, "requests": 10}
    ]

def normalize_benchmark(b):
    mult = {"FP32": 0.5, "FP16": 1.0, "INT8": 1.5, "INT4": 2.0}
    p = b["precision"]
    factor = mult.get(p, 1.0)
    norm_tps = b["raw_tps"] * factor
    per_gpu = norm_tps / b["gpus"]
    return per_gpu

def audit_benchmark(b):
    issues = []
    if b.get("precision") not in ("FP16",):
        issues.append("quantization_mismatch")
    if not b.get("warmup", True):
        issues.append("no_warmup")
    if b.get("requests", 10) <= 1:
        issues.append("single_request")
    return issues

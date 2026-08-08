import numpy as np

def analyze_k_sweep(records):
    sweps = {}
    for r in records:
        k = r.get("k_val", 0)
        kernel = r.get("kernel", "unknown")
        sweps.setdefault(k, []).append(kernel)
    result = {}
    for k, kernels in sweps.items():
        counts = {"avx2": 0, "avx512": 0, "amx": 0}
        for kr in kernels:
            if kr in counts:
                counts[kr] += 1
        dominant = max(counts, key=counts.get)
        result[k] = dominant
    return result

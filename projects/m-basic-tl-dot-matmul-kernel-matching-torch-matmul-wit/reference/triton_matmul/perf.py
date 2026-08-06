def analyze_tflops(benchmark_data, m, n, k):
    ops = 2.0 * m * n * k
    found = None
    for row in benchmark_data:
        if row.get("m") == m and row.get("n") == n and row.get("k") == k:
            found = row
            break
    if found is None:
        return {"tflops": 0.0, "cublas_tflops": 0.0, "ratio": 0.0}
    time_ms = found.get("time_ms", 1.0)
    tflops = ops / (time_ms * 1e-3) / 1e12
    cublas_time_ms = found.get("cublas_time_ms", time_ms)
    cublas_tflops = ops / (cublas_time_ms * 1e-3) / 1e12
    ratio = tflops / cublas_tflops if cublas_tflops > 0 else 0.0
    return {"tflops": tflops, "cublas_tflops": cublas_tflops, "ratio": ratio}
